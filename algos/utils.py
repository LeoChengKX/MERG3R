import copy
import gc
import os
import glob
import shutil
import tempfile
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import trimesh
import requests
from PIL import Image
from scipy.spatial.transform import Rotation

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map
from pi3.models.pi3 import Pi3


def extrinsic_to_colmap_format(extrinsics):
    """Convert extrinsic matrices to COLMAP format (quaternion + translation)."""
    num_cameras = extrinsics.shape[0]
    quaternions = []
    translations = []

    for i in range(num_cameras):
        # VGGT's extrinsic is camera-to-world (R|t) format
        R = extrinsics[i, :3, :3]
        t = extrinsics[i, :3, 3]

        # Convert rotation matrix to quaternion
        # COLMAP quaternion format is [qw, qx, qy, qz]
        rot = Rotation.from_matrix(R)
        quat = rot.as_quat()  # scipy returns [x, y, z, w]
        quat = np.array([quat[3], quat[0], quat[1], quat[2]])  # Convert to [w, x, y, z]

        quaternions.append(quat)
        translations.append(t)

    return np.array(quaternions), np.array(translations)


def _download_file_from_url(url, filename):
    """Downloads a file from a URL, handling redirects."""
    try:
        response = requests.get(url, allow_redirects=False)
        response.raise_for_status()

        if response.status_code == 302:
            redirect_url = response.headers["Location"]
            response = requests.get(redirect_url, stream=True)
            response.raise_for_status()
        else:
            response = requests.get(url, stream=True)
            response.raise_for_status()

        with open(filename, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Downloaded {filename} successfully.")
        return True

    except requests.exceptions.RequestException as e:
        print(f"Error downloading file: {e}")
        return False


def _run_skyseg(onnx_session, input_size, image):
    """Runs sky segmentation inference using ONNX model."""
    temp_image = copy.deepcopy(image)
    resize_image = cv2.resize(temp_image, dsize=(input_size[0], input_size[1]))
    x = cv2.cvtColor(resize_image, cv2.COLOR_BGR2RGB)
    x = np.array(x, dtype=np.float32)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    x = (x / 255 - mean) / std
    x = x.transpose(2, 0, 1)
    x = x.reshape(-1, 3, input_size[0], input_size[1]).astype("float32")

    input_name = onnx_session.get_inputs()[0].name
    output_name = onnx_session.get_outputs()[0].name
    onnx_result = onnx_session.run([output_name], {input_name: x})

    onnx_result = np.array(onnx_result).squeeze()
    min_value = np.min(onnx_result)
    max_value = np.max(onnx_result)
    onnx_result = (onnx_result - min_value) / (max_value - min_value)
    onnx_result *= 255
    onnx_result = onnx_result.astype("uint8")

    return onnx_result


def _segment_sky(image_path, onnx_session, mask_filename=None):
    """Segments sky from an image using an ONNX model."""
    image = cv2.imread(image_path)

    result_map = _run_skyseg(onnx_session, [320, 320], image)
    result_map_original = cv2.resize(result_map, (image.shape[1], image.shape[0]))

    output_mask = np.zeros_like(result_map_original)
    output_mask[result_map_original < 32] = 255

    if mask_filename is not None:
        os.makedirs(os.path.dirname(mask_filename), exist_ok=True)
        cv2.imwrite(mask_filename, output_mask)

    return output_mask


def _hash_point(point, scale=100):
    """Create a hash for a 3D point by quantizing coordinates."""
    quantized = tuple(np.round(point * scale).astype(int))
    return hash(quantized)


def filter_and_prepare_points(
    predictions,
    conf_threshold,
    mask_sky=False,
    mask_black_bg=False,
    mask_white_bg=False,
    stride=1,
    prediction_mode="Depthmap and Camera Branch",
):
    """
    Filter points based on confidence and prepare for COLMAP format.
    Implementation matches the conventions in the original VGGT code.
    """
    if "Pointmap" in prediction_mode:
        print("Using Pointmap Branch")
        if "world_points" in predictions:
            pred_world_points = predictions["world_points"]
            pred_world_points_conf = predictions.get(
                "world_points_conf", np.ones_like(pred_world_points[..., 0])
            )
        else:
            print(
                "Warning: world_points not found in predictions, falling back to depth-based points"
            )
            pred_world_points = predictions["world_points_from_depth"]
            pred_world_points_conf = predictions.get(
                "depth_conf", np.ones_like(pred_world_points[..., 0])
            )
    else:
        print("Using Depthmap and Camera Branch")
        pred_world_points = predictions["world_points_from_depth"]
        pred_world_points_conf = predictions.get(
            "depth_conf", np.ones_like(pred_world_points[..., 0])
        )

    if "colmap_images" in predictions.keys():
        colors_rgb = predictions["colmap_images"]
    else:
        colors_rgb = predictions["images"]

    S, H, W = pred_world_points.shape[:3]
    if colors_rgb.shape[:3] != (S, H, W):
        print(f"Reshaping colors_rgb from {colors_rgb.shape} to match {(S, H, W, 3)}")
        reshaped_colors = np.zeros((S, H, W, 3), dtype=np.float32)
        for i in range(S):
            if i < len(colors_rgb):
                reshaped_colors[i] = cv2.resize(colors_rgb[i], (W, H))
        colors_rgb = reshaped_colors

    colors_rgb = (colors_rgb * 255).astype(np.uint8)

    if mask_sky:
        print("Applying sky segmentation mask")
        try:
            import onnxruntime

            with tempfile.TemporaryDirectory() as temp_dir:
                print(f"Created temporary directory for sky segmentation: {temp_dir}")
                temp_images_dir = os.path.join(temp_dir, "images")
                sky_masks_dir = os.path.join(temp_dir, "sky_masks")
                os.makedirs(temp_images_dir, exist_ok=True)
                os.makedirs(sky_masks_dir, exist_ok=True)

                image_list = []
                for i, img in enumerate(colors_rgb):
                    img_path = os.path.join(temp_images_dir, f"image_{i:04d}.png")
                    image_list.append(img_path)
                    cv2.imwrite(img_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

                skyseg_path = os.path.join(temp_dir, "skyseg.onnx")
                if not os.path.exists("skyseg.onnx"):
                    print("Downloading skyseg.onnx...")
                    download_success = _download_file_from_url(
                        "https://huggingface.co/JianyuanWang/skyseg/resolve/main/skyseg.onnx",
                        skyseg_path,
                    )
                    if not download_success:
                        print("Failed to download skyseg model, skipping sky filtering")
                        mask_sky = False
                else:
                    shutil.copy("skyseg.onnx", skyseg_path)

                if mask_sky:
                    skyseg_session = onnxruntime.InferenceSession(skyseg_path)
                    sky_mask_list = []

                    for img_path in image_list:
                        mask_path = os.path.join(
                            sky_masks_dir, os.path.basename(img_path)
                        )
                        sky_mask = _segment_sky(img_path, skyseg_session, mask_path)

                        if sky_mask.shape[0] != H or sky_mask.shape[1] != W:
                            sky_mask = cv2.resize(sky_mask, (W, H))

                        sky_mask_list.append(sky_mask)

                    sky_mask_array = np.array(sky_mask_list)
                    sky_mask_binary = (sky_mask_array > 0.1).astype(np.float32)
                    pred_world_points_conf = pred_world_points_conf * sky_mask_binary
                    print(f"Applied sky mask, shape: {sky_mask_binary.shape}")

        except (ImportError, Exception) as e:
            print(f"Error in sky segmentation: {e}")
            mask_sky = False

    vertices_3d = pred_world_points.reshape(-1, 3)
    conf = pred_world_points_conf.reshape(-1)
    colors_rgb_flat = colors_rgb.reshape(-1, 3)

    if len(conf) != len(colors_rgb_flat):
        print(
            f"WARNING: Shape mismatch between confidence ({len(conf)}) and colors ({len(colors_rgb_flat)})"
        )
        min_size = min(len(conf), len(colors_rgb_flat))
        conf = conf[:min_size]
        vertices_3d = vertices_3d[:min_size]
        colors_rgb_flat = colors_rgb_flat[:min_size]

    if conf_threshold == 0.0:
        conf_thres_value = 0.0
    else:
        conf_thres_value = np.percentile(conf, conf_threshold)

    print(
        f"Using confidence threshold: {conf_threshold}% (value: {conf_thres_value:.4f})"
    )
    conf_mask = (conf >= conf_thres_value) & (conf > 1e-5)

    if mask_black_bg:
        print("Filtering black background")
        black_bg_mask = colors_rgb_flat.sum(axis=1) >= 16
        conf_mask = conf_mask & black_bg_mask

    if mask_white_bg:
        print("Filtering white background")
        white_bg_mask = ~(
            (colors_rgb_flat[:, 0] > 240)
            & (colors_rgb_flat[:, 1] > 240)
            & (colors_rgb_flat[:, 2] > 240)
        )
        conf_mask = conf_mask & white_bg_mask

    filtered_vertices = vertices_3d[conf_mask]
    filtered_colors = colors_rgb_flat[conf_mask]

    if len(filtered_vertices) == 0:
        print("Warning: No points remaining after filtering. Using default point.")
        filtered_vertices = np.array([[0, 0, 0]])
        filtered_colors = np.array([[200, 200, 200]])

    print(f"Filtered to {len(filtered_vertices)} points")

    points3D = []
    point_indices = {}
    image_points2D = [[] for _ in range(len(pred_world_points))]

    print(f"Preparing points for COLMAP format with stride {stride}...")

    total_points = 0
    for img_idx in range(S):
        for y in range(0, H, stride):
            for x in range(0, W, stride):
                flat_idx = img_idx * H * W + y * W + x

                if flat_idx >= len(conf):
                    continue

                if conf[flat_idx] < conf_thres_value or conf[flat_idx] <= 1e-5:
                    continue

                if mask_black_bg and colors_rgb_flat[flat_idx].sum() < 16:
                    continue

                if mask_white_bg and all(colors_rgb_flat[flat_idx] > 240):
                    continue

                point3D = vertices_3d[flat_idx]
                rgb = colors_rgb_flat[flat_idx]

                if not np.all(np.isfinite(point3D)):
                    continue

                point_hash = _hash_point(point3D, scale=100)

                if point_hash not in point_indices:
                    point_idx = len(points3D)
                    point_indices[point_hash] = point_idx

                    point_entry = {
                        "id": point_idx,
                        "xyz": point3D,
                        "rgb": rgb,
                        "error": 1.0,
                        "track": [(img_idx, len(image_points2D[img_idx]))],
                    }
                    points3D.append(point_entry)
                    total_points += 1
                else:
                    point_idx = point_indices[point_hash]
                    points3D[point_idx]["track"].append(
                        (img_idx, len(image_points2D[img_idx]))
                    )

                image_points2D[img_idx].append(
                    (x, y, point_indices[point_hash])
                )

    print(
        f"Prepared {len(points3D)} 3D points with {sum(len(pts) for pts in image_points2D)} observations for COLMAP"
    )
    return points3D, image_points2D


def write_colmap_cameras_txt(file_path, intrinsics, image_width, image_height):
    """Write camera intrinsics to COLMAP cameras.txt format."""
    with open(file_path, "w") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"# Number of cameras: {len(intrinsics)}\n")

        for i, intrinsic in enumerate(intrinsics):
            camera_id = i + 1
            model = "PINHOLE"

            fx = intrinsic[0, 0]
            fy = intrinsic[1, 1]
            cx = intrinsic[0, 2]
            cy = intrinsic[1, 2]

            f.write(
                f"{camera_id} {model} {image_width} {image_height} {fx} {fy} {cx} {cy}\n"
            )


def write_colmap_images_txt(
    file_path, quaternions, translations, image_points2D, image_names
):
    """Write camera poses and keypoints to COLMAP images.txt format."""
    with open(file_path, "w") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")

        num_points = sum(len(points) for points in image_points2D)
        avg_points = num_points / len(image_points2D) if image_points2D else 0
        f.write(
            f"# Number of images: {len(quaternions)}, mean observations per image: {avg_points:.1f}\n"
        )

        for i in range(len(quaternions)):
            image_id = i + 1
            camera_id = i + 1

            qw, qx, qy, qz = quaternions[i]
            tx, ty, tz = translations[i]

            f.write(
                f"{image_id} {qw} {qx} {qy} {qz} {tx} {ty} {tz} {camera_id} {os.path.basename(image_names[i])}\n"
            )

            points_line = " ".join(
                [
                    f"{x} {y} {point3d_id+1}"
                    for x, y, point3d_id in image_points2D[i]
                ]
            )
            f.write(f"{points_line}\n")


def write_colmap_points3D_txt(file_path, points3D):
    """Write 3D points and tracks to COLMAP points3D.txt format."""
    with open(file_path, "w") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write(
            "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n"
        )

        avg_track_length = (
            sum(len(point["track"]) for point in points3D) / len(points3D)
            if points3D
            else 0
        )
        f.write(
            f"# Number of points: {len(points3D)}, mean track length: {avg_track_length:.4f}\n"
        )

        for point in points3D:
            point_id = point["id"] + 1
            x, y, z = point["xyz"]
            r, g, b = point["rgb"]
            error = point["error"]

            track = " ".join(
                f"{img_id+1} {point2d_idx}" for img_id, point2d_idx in point["track"]
            )

            f.write(
                f"{point_id} {x} {y} {z} {int(r)} {int(g)} {int(b)} {error} {track}\n"
            )



def load_model(model_name="vggt", device=None):
    """Load and initialize the VGGT model."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    if model_name == 'vggt':
        model = VGGT.from_pretrained("facebook/VGGT-1B")
    elif model_name == 'pi3':
        model = Pi3.from_pretrained("yyfz233/Pi3")
    else:
        raise NotImplementedError("Other model backbones are not implemented!")

    
    model.eval()
    model = model.to(device)
    return model, device



def run_inference_step_by_step(model, batches, size_hw, device, need_features=False):
    """
    Output:
     - extrinsic: (N, 3, 4)
     - intrinsic: (N, 3, 3)
     - depth: (N, H, W, 1)
     - depth_conf: (N, H, W)
    """
    # Construct a customized prediction
    predictions = []
    start = time.time()

    if isinstance(model, VGGT):
        for i, images in enumerate(batches):
            prediction = dict()

            with torch.no_grad():
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    images = images[None].to(device)  # add batch dimension
                    aggregated_tokens_list, ps_idx, patch_tokens = model.aggregator(images)
            
                    pose_enc = model.camera_head(aggregated_tokens_list)[-1]
                    # Predict depth maps
                    depth_map, depth_conf = model.depth_head(aggregated_tokens_list, images, ps_idx)
                    # No need to predict point maps here
                    # Predict feature maps for tracking
                    if need_features:
                        feature_maps = model.track_head.feature_extractor(aggregated_tokens_list, images, ps_idx)

            prediction['depth'] = depth_map.to(dtype=torch.float32, device='cpu')
            prediction['depth_conf'] = depth_conf.to(dtype=torch.float32, device='cpu')
            prediction['pose_enc'] = pose_enc.to(dtype=torch.float32, device='cpu')

            extri, intri = pose_encoding_to_extri_intri(prediction["pose_enc"], size_hw)
            prediction['extrinsic'] = extri.squeeze(0)
            prediction['intrinsic'] = intri.squeeze(0)
            prediction['world_points'] = unproject_depth_map_to_point_map(prediction['depth'].squeeze(0), extri.squeeze(0), intri.squeeze(0))

            if need_features:
                prediction['features'] = feature_maps.to(dtype=torch.float32, device='cpu')
                
            for key in patch_tokens.keys():
                if isinstance(patch_tokens[key], torch.Tensor):
                    val = patch_tokens[key].to(dtype=torch.float32, device='cpu')
                    patch_tokens[key] = val
            # prediction['dino_features'] = patch_tokens.to("cpu")

            del aggregated_tokens_list, ps_idx, patch_tokens, pose_enc, depth_map, depth_conf, images

            predictions.append(prediction)
            torch.cuda.empty_cache()
            gc.collect()
        
    elif isinstance(model, Pi3):
        for i, images in enumerate(batches):
            prediction = dict()
            with torch.no_grad():
                    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                        images = images[None].to(device)  # add batch dimension
                        res = model(images)
            
            prediction['extrinsic'] = remove_homogeneous_row(torch.linalg.inv(res['camera_poses'].squeeze(0))).to(dtype=torch.float32, device='cpu')
            prediction['world_points'] = res['points'].to(dtype=torch.float32, device='cpu').squeeze(0)
            prediction['depth_conf'] = res['conf'].to(dtype=torch.float32, device='cpu').squeeze(-1)
            prediction['depth_conf'] = torch.sigmoid(prediction['depth_conf'])
            # prediction['dino_features'] = res['dino_features']

            intrinsic, depth = estimate_intrinsics_and_depth(res['local_points'].squeeze(0))
            prediction['intrinsic'] = intrinsic.to(dtype=torch.float32, device='cpu').squeeze(0)
            prediction['depth'] = compute_depth(prediction['world_points'], prediction['extrinsic']).to(dtype=torch.float32, device='cpu').unsqueeze(0).unsqueeze(-1)

            predictions.append(prediction)

            del res, images
            
            gc.collect()
            torch.cuda.empty_cache()
            

    end = time.time()

    print(f"[INFERENCE] Time used: {end - start}s. ")
    
    return predictions


def compute_depth(points, extrin):
    """
    points: (N, H, W, 3) in world coordinates
    extrin: (N, 3, 4) camera extrinsics (world -> camera)
    intrin: (N, 3, 3) camera intrinsics (unused for depth directly)
    
    Returns:
        depth: (N, H, W)
    """
    N, H, W, _ = points.shape

    # Convert to homogeneous coords: (N, H, W, 4)
    ones = torch.ones((N, H, W, 1), dtype=points.dtype, device=points.device)
    homog_points = torch.cat([points, ones], dim=-1)

    # Transform world -> camera: (N, H, W, 3)
    # First expand extrin to match dimensions
    cam_points = torch.einsum('nij,nhwj->nhwi', extrin, homog_points)

    # Depth is the z-coordinate in camera space
    depth = cam_points[..., 2]

    return depth


@torch.no_grad()
def estimate_intrinsics_and_depth(points: torch.Tensor):
    """
    Estimate intrinsics K (fx, fy, cx, cy, zero skew) from points in camera coords,
    and also return per-pixel depths.

    Args:
        points: (B, H, W, 3)  3D points expressed in the CAMERA frame
                              (i = row = y, j = col = x).
                              Last dimension = (X, Y, Z).

    Returns:
        K:      (B, 3, 3) intrinsics matrix for each camera
        depth:  (B, H, W) depth = Z coordinate (positive forward)
    """
    B, H, W, _ = points.shape
    device = points.device
    # Pixel grids: u = x (cols), v = y (rows)
    u_grid = torch.arange(W, device=device, dtype=torch.float).view(1, 1, W).expand(B, H, W)
    v_grid = torch.arange(H, device=device, dtype=torch.float).view(1, H, 1).expand(B, H, W)

    X = points[..., 0]
    Y = points[..., 1]
    Z = points[..., 2]   # depth map

    # Valid mask
    valid = torch.isfinite(Z) & (Z > 1e-6)

    # Prepare outputs
    K = torch.zeros((B, 3, 3), dtype=points.dtype, device=device)
    K[:, 2, 2] = 1.0

    for b in range(B):
        m = valid[b]
        if m.sum().item() < 4:
            K[b] = torch.full((3, 3), float("nan"), dtype=points.dtype, device=device)
            K[b, 2, 2] = 1.0
            continue

        # Build linear systems: u = fx * (X/Z) + cx ; v = fy * (Y/Z) + cy
        a_u = (X[b][m] / Z[b][m]).unsqueeze(1)
        a_v = (Y[b][m] / Z[b][m]).unsqueeze(1)
        A_u = torch.cat([a_u, torch.ones_like(a_u)], dim=1)
        A_v = torch.cat([a_v, torch.ones_like(a_v)], dim=1)

        u = u_grid[b][m].unsqueeze(1)
        v = v_grid[b][m].unsqueeze(1)

        sol_u = torch.linalg.lstsq(A_u, u).solution.squeeze(1)
        sol_v = torch.linalg.lstsq(A_v, v).solution.squeeze(1)

        fx, cx = sol_u[0], sol_u[1]
        fy, cy = sol_v[0], sol_v[1]

        K[b, 0, 0] = fx
        K[b, 1, 1] = fy
        K[b, 0, 2] = cx
        K[b, 1, 2] = cy

    return K, Z   # Z is the depth map



def convert_to_homogeneous_matrix(input: torch.Tensor):
    if len(input.shape) == 2:
        # No batch dimension
        return torch.cat([input, torch.tensor([0, 0, 0, 1]).reshape(1, 4).to(input.device)])

    elif len(input.shape) == 3:
        # With batch dimension
        bs = input.shape[0]
        homo_part = torch.stack([torch.tensor([0, 0, 0, 1]).reshape(1, 4) for _ in range(bs)]).to(input.device)
        return torch.cat([input, homo_part], dim=1)

    else:
        raise ValueError("Input shape incorrect for homogeneous matrix. ")


def remove_homogeneous_row(matrix: torch.Tensor) -> torch.Tensor:
    if len(matrix.shape) == 2:
        # No batch dimension, shape should be (4, 4)
        if matrix.shape != (4, 4):
            raise ValueError("Expected shape (4, 4) for single matrix.")
        return matrix[:3]

    elif len(matrix.shape) == 3:
        # With batch dimension, shape should be (B, 4, 4)
        if matrix.shape[1:] != (4, 4):
            raise ValueError("Expected shape (B, 4, 4) for batched matrix.")
        return matrix[:, :3]

    else:
        raise ValueError("Invalid input shape.")
    

def output_to_colmap(predictions, img_names, output_dir, image_points2D, points3D, idx=None):
    name = "colmap" if idx is None else f"colmap_{idx}"
    quaternions, translations = extrinsic_to_colmap_format(predictions['extrinsic'])
    
    height, width = predictions["depth"].shape[1:3]
    os.makedirs(os.path.join(output_dir, name), exist_ok=True)
    write_colmap_cameras_txt(
        os.path.join(output_dir, name, "cameras.txt"), 
        predictions["intrinsic"], width, height)
    write_colmap_images_txt(
        os.path.join(output_dir, name, "images.txt"), 
        quaternions, translations, image_points2D, img_names)
    write_colmap_points3D_txt(
        os.path.join(output_dir, name, "points3D.txt"), 
        points3D)


def write_recon_to_colmap(output_dir, predictions, images, names, stride=100, conf_threshold=50.0):
    size_hw = images.shape[-2:]
    if "extrinsic" not in predictions.keys():
        extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], size_hw)
        predictions["extrinsic"] = extrinsic
    if "intrinsic" not in predictions.keys():
        extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], size_hw)
        predictions["intrinsic"] = intrinsic
    
    for key in predictions.keys():
        if isinstance(predictions[key], torch.Tensor):
            predictions[key] = predictions[key].cpu().numpy() 
            if predictions[key].shape[0] == 1:
                predictions[key] = predictions[key].squeeze(0) # remove batch dimension
    
    predictions["original_images"] = images
    print("[OUTPUT WRITING] Computing 3D points from depth maps...")
    depth_map = predictions["depth"]  # (S, H, W, 1)
    world_points = unproject_depth_map_to_point_map(depth_map, predictions["extrinsic"], predictions["intrinsic"])
    predictions["world_points_from_depth"] = world_points

    S, H, W = world_points.shape[:3]
    normalized_images = np.zeros((S, H, W, 3), dtype=np.float32)
    for j, img in enumerate(images):
        img = img.permute(1, 2, 0)
        resized_img = cv2.resize(img.cpu().numpy(), (W, H))
        # normalized_images[j] = resized_img / 255.0
        normalized_images[j] = resized_img
    
    predictions["colmap_images"] = normalized_images

    print(f"[OUTPUT WRITING] Filtering points with confidence threshold {conf_threshold}% and stride {stride}...")
    points3D, image_points2D = filter_and_prepare_points(
        predictions, 
        conf_threshold, 
        mask_sky=False, 
        mask_black_bg=False,
        mask_white_bg=False,
        stride=stride,
        prediction_mode="Depthmap and Camera Branch"
    )
    
    # Export 3D point clouds
    points_3d = []
    points_rgb = []
    for point in points3D:
        point_id = point["id"] + 1  
        x, y, z = point["xyz"]
        r, g, b = point["rgb"]

        points_3d.append((x, y, z))
        points_rgb.append((r, g, b, 1))
    trimesh.PointCloud(points_3d, points_rgb).export(os.path.join(output_dir, "points.ply"))
    output_to_colmap(predictions, names, output_dir, image_points2D, points3D)


def restore_predictions_order(predictions):
    """
    Sort the prediction based on image_ids to restore the order of all the attributes
    to the original order.
    """
    image_ids = predictions['image_ids']

    for key in predictions.keys():
        if isinstance(predictions[key], np.ndarray):
            rearrangement = np.argsort(image_ids)
            if predictions[key].shape[0] == rearrangement.shape[0]:
                predictions[key] = predictions[key][rearrangement]


def extract_frames_from_video(video_path, subsample=1, num_images=-1, output_dir=None):
    """
    Extract frames from a video file and save them as images.
    
    Args:
        video_path: Path to the video file
        subsample: Extract every Nth frame (default: 1, extract all frames)
        num_images: Maximum number of frames to extract (-1 for all frames)
        output_dir: Directory to save frames. If None, creates a temporary directory.
    
    Returns:
        tuple: (list of frame file paths, output directory path)
    """
    print(f"[UTILS] Processing video file: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")
    
    # Create output directory if not provided
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="video_frames_")
    else:
        os.makedirs(output_dir, exist_ok=True)
    
    frame_idx = 0
    saved_frame_idx = 0
    image_names = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Apply subsampling
        if frame_idx % subsample == 0:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Save frame as image
            frame_path = os.path.join(output_dir, f"frame_{saved_frame_idx:06d}.jpg")
            cv2.imwrite(frame_path, cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR))
            image_names.append(frame_path)
            saved_frame_idx += 1
            
            # Apply num_images limit if specified
            if num_images != -1 and len(image_names) >= num_images:
                break
        
        frame_idx += 1
    
    cap.release()
    
    if len(image_names) == 0:
        if output_dir is None or output_dir.startswith(tempfile.gettempdir()):
            shutil.rmtree(output_dir, ignore_errors=True)
        raise ValueError(f"No frames extracted from video: {video_path}")
    
    print(f"[UTILS] Extracted {len(image_names)} frames from video")
    return image_names, output_dir


def process_images(image_dir, subsample, device, num_images, multi_dirs=False):
    """Process images with VGGT and return predictions. Also supports video files."""
    
    # Check if input is a video file
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.webm', '.MP4', '.AVI', '.MOV', '.MKV', '.WEBM')
    is_video = os.path.isfile(image_dir) and image_dir.lower().endswith(video_extensions)
    
    if is_video:
        # Extract frames from video
        image_names, _ = extract_frames_from_video(image_dir, subsample, num_images)
    else:
        # Original image directory processing
        if multi_dirs:
            image_names = glob.glob(os.path.join(image_dir, "*", "**"), recursive=True)
        else:
            image_names = glob.glob(os.path.join(image_dir, "*"))
        image_names = sorted([f for f in image_names if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        if num_images != -1:
            image_names = image_names[:num_images]
        image_names = image_names[::subsample] # subsampling code from mast3r-sfm
        print(f"[UTILS] Found {len(image_names)} images")
        
        if len(image_names) == 0:
            raise ValueError(f"No images found in {image_dir}")

    original_images = []
    for img_path in image_names:
        img = Image.open(img_path).convert('RGB')
        original_images.append(np.array(img))
    
    images = load_and_preprocess_images(image_names).to(device)

    print(f"[UTILS] Preprocessed images shape: {images.shape}")
    
    return images, image_names
    

@torch.no_grad()
def mnn_one_to_many(A, B, tau=0.6):
    """
    A: (P, D)  normalized patch tokens (image i)
    B: (Bsz, P, D) normalized patch tokens (candidate images)
    tau: cosine threshold

    Returns:
        scores: (Bsz,) MNN score for each candidate in B
    """
    # Ensure correct dims
    if B.dim() == 2:
        B = B.unsqueeze(0)  # (1, P, D)

    # (Bsz, P, P): S[b, i, j] = <B[b,i], A[j]>
    # This matches each patch in B to all patches in A.
    S = torch.matmul(B, A.t())  # (Bsz, P, P)

    # For each patch i in B: best j in A
    j_best = S.argmax(dim=2)  # (Bsz, P)

    # For each patch j in A: best i in B
    i_best = S.argmax(dim=1)  # (Bsz, P)

    P = A.shape[0]
    i_idx = torch.arange(P, device=A.device).view(1, P).expand(B.shape[0], P)  # (Bsz, P)

    # mutual: i_best[b, j_best[b,i]] == i
    i_back = torch.gather(i_best, dim=1, index=j_best)  # (Bsz, P)
    mutual = (i_back == i_idx)

    # confidence: S[b, i, j_best[b,i]] > tau
    sim_ij = torch.gather(S, dim=2, index=j_best.unsqueeze(2)).squeeze(2)  # (Bsz, P)
    confident = (sim_ij > tau)

    good = mutual & confident
    return good.float().mean(dim=1)  # (Bsz,)      # (Bsz,)


@torch.no_grad()
def mnn_from_dino_candidates(
    X, sim_dino, K=30, tau=0.6, batch_cand=8, use_fp16=True, symmetric=True
):
    """
    X: (M, P, D) patch tokens
    sim_dino: (M, M) DINO similarity matrix (larger = more similar)
    K: number of candidates per image to verify with MNN
    tau: patch-level cosine threshold for confident MNN matches
    batch_cand: compute MNN for this many candidates at once
    symmetric: mirror scores to make S_mnn symmetric

    Returns:
        S_mnn: (M, M) float32 matrix with only candidate entries filled (others 0)
        cand_idx: (M, K) candidate indices used per row
    """
    device = X.device
    M, P, D = X.shape

    # Normalize patch tokens for cosine
    X = F.normalize(X, p=2, dim=-1)

    # Make sure diagonal doesn't get selected
    sim = sim_dino.clone()
    sim.fill_diagonal_(-1e9)

    # Candidate indices (top-K per row)
    cand_vals, cand_idx = torch.topk(sim, k=K, dim=1, largest=True, sorted=True)  # (M,K)

    S_mnn = torch.zeros((M, M), device=device, dtype=torch.float32)

    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if (use_fp16 and device.type == "cuda")
        else torch.cpu.amp.autocast(enabled=False)
    )

    for i in range(M):
        A = X[i]  # (P,D)
        js = cand_idx[i]  # (K,)

        # optional: if symmetric, only compute j>i to halve work
        if symmetric:
            js = js[js > i]
            if js.numel() == 0:
                continue

        # process candidates in batches
        for t in range(0, js.numel(), batch_cand):
            j_batch = js[t:t+batch_cand]
            B = X[j_batch]  # (Bsz,P,D)

            with autocast_ctx:
                scores = mnn_one_to_many(A, B, tau=tau)  # (Bsz,)

            S_mnn[i, j_batch] = scores.float()

    if symmetric:
        S_mnn = torch.maximum(S_mnn, S_mnn.t())  # keep larger of two directions
        S_mnn.fill_diagonal_(0.0)

    return S_mnn, cand_idx


def get_sim_matrix(
    images,
    model_name="dinov3",
    device="cuda",
    subset_size=100,
    feature_size=768,
    alpha=0.3,
    return_feats=False
):
        """Extract normalized feature embeddings for all images.

        Loads a DINOv2/DINOv3 model from Torch Hub, normalizes images using
        ImageNet statistics, and obtains per-frame patch embeddings.
        Similarity is computed as a blend of:
        - Global mean-patch cosine similarity (diagonal zeroed).
        - MNN (mutual nearest-neighbor) patch consistency.

        """
        _RESNET_MEAN = [0.485, 0.456, 0.406]
        _RESNET_STD = [0.229, 0.224, 0.225]
        if model_name == "dinov3":
            repo_dir = os.environ.get("MERG3R_DINOV3_REPO_DIR")
            weights_url = os.environ.get("MERG3R_DINOV3_WEIGHTS_URL", None)
            if not repo_dir:
                raise ValueError(
                    "DINOv3 requires MERG3R_DINOV3_REPO_DIR (path to dinov3 repo). "
                    "Use model_name='dinov2' for the default public model."
                )
            model = torch.hub.load(
                repo_dir, "dinov3_vitb16", source="local", weights=weights_url
            )
        else:
            model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14_reg")

        model.eval()
        model = model.to(device)
        
        resnet_mean = torch.tensor(_RESNET_MEAN, device=device).view(1, 3, 1, 1)
        resnet_std = torch.tensor(_RESNET_STD, device=device).view(1, 3, 1, 1)
        images_resnet_norm = (images - resnet_mean) / resnet_std
        
        # Need to split up into subsets because of feature extractor GPU limitations
        num_subsets = (len(images) // subset_size) + 1
        frame_feat = torch.empty(size=(0, feature_size), device=device)
        frame_chunks = []
        with torch.no_grad():
            for i in range(num_subsets):
                image_subset = images_resnet_norm[i * subset_size : (i+1) * subset_size]
                frame_feat_subset = model(image_subset, is_training=True)
                # frame_feat_subset = frame_feat_subset["x_norm_clstoken"]

                # --- Mean pooling patches
                frame_feat_subset = frame_feat_subset["x_norm_patchtokens"] 
                # frame_feat_subset = torch.mean(frame_feat_subset, dim=1)

                frame_chunks.append(frame_feat_subset)
                del image_subset, frame_feat_subset
        frame_feat = torch.cat(frame_chunks, dim=0)
        frame_feat_norm = torch.nn.functional.normalize(frame_feat, p=2, dim=-1)
        frame_feat_norm_mean = torch.mean(frame_feat_norm, dim=1)
        
        del model, resnet_mean, resnet_std, images_resnet_norm
        sim_matrix = frame_feat_norm_mean @ frame_feat_norm_mean.t()
        sim_matrix = sim_matrix.fill_diagonal_(0)

        mnn_sim_matrix, cand_idx = mnn_from_dino_candidates(frame_feat_norm, sim_matrix)
        sim_matrix = alpha * sim_matrix + (1 - alpha) * mnn_sim_matrix

        gc.collect()
        torch.cuda.empty_cache()
        if return_feats:
            return sim_matrix, frame_feat_norm_mean
        return sim_matrix



def rbd(data: dict) -> dict:
    """Remove batch dimension from elements in data"""
    return {
        k: v[0] if isinstance(v, (torch.Tensor, np.ndarray, list)) else v
        for k, v in data.items()
    }