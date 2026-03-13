"""
PyTorch version of depth maps point maps conversion
"""
import torch
import numpy as np

def project_3d_points_to_image(
        points: torch.Tensor, rotation: torch.Tensor, translation: torch.Tensor, intrinsic: torch.Tensor
    ):
    """
    Project the points using extrinsic and intrinsic to the image plane. 

    points: (P, 3) in world coordinate
    extrinsic:
      - rotation (3, 3)
      - translation (3, 1)
    intrinsic: (3, 3)

    return: 2d points (P, 2)
    """
    points_camera = rotation @ points.T + translation  # (3, P)

    points_img = intrinsic @ points_camera

    # Normalize the pixel coordinates
    z_coord = points_img[2, :].clip(1e-3)
    x = points_img[0, :] / z_coord
    y = points_img[1, :] / z_coord
    pixel_coords = torch.stack([x, y], dim=1)  # (P, 2)

    # Mask: valid only if in front of camera
    valid_mask = points_camera[2, :] > 0

    return pixel_coords, valid_mask


def project_3d_points_to_image_numpy(
        points: np.ndarray, rotation: np.ndarray, translation: np.ndarray, intrinsic: np.ndarray
    ):
    """
    Project the points using extrinsic and intrinsic to the image plane. 

    points: (P, 3) in world coordinate
    extrinsic:
      - rotation (3, 3)
      - translation (3, 1)
    intrinsic: (3, 3)

    return: 2d points (P, 2)
    """
    points_camera = rotation @ points.T + translation  # (3, P)

    points_img = intrinsic @ points_camera

    # Normalize the pixel coordinates
    z_coord = points_img[2, :].clip(1e-3)
    x = points_img[0, :] / z_coord
    y = points_img[1, :] / z_coord
    pixel_coords = np.stack([x, y], axis=1)  # (P, 2)

    # Mask: valid only if in front of camera
    valid_mask = points_camera[2, :] > 0

    return pixel_coords, valid_mask


def project_3d_points_to_image_torch(
        points: torch.Tensor, rotation: torch.Tensor, translation: torch.Tensor, intrinsic: torch.Tensor
    ):
    """
    Project the points using extrinsic and intrinsic to the image plane. 

    points: (P, 3) in world coordinate
    extrinsic:
      - rotation (3, 3)
      - translation (3, 1)
    intrinsic: (3, 3)

    return: 2d points (P, 2)
    """
    points_camera = rotation @ points.T + translation  # (3, P)

    points_img = intrinsic @ points_camera   # (3, P)

    # Normalize the pixel coordinates
    z_coord = points_img[2:3, :].clip(1e-3)
    xy = points_img[:2, :] / z_coord
    pixel_coords = xy.transpose(0, 1)

    # Mask: valid only if in front of camera
    valid_mask = points_camera[2, :] > 0

    return pixel_coords, valid_mask


def project_3d_points_to_image_batch(
        points: torch.Tensor, rotation: torch.Tensor, translation: torch.Tensor, intrinsic: torch.Tensor
    ):
    """
    Project the points using extrinsic and intrinsic to the image plane. 

    points: (N, P, 3) in world coordinate
    extrinsic:
      - rotation (N, 3, 3)
      - translation (N, 3, 1)
    intrinsic: (N, 3, 3)

    return: 2d points (N, P, 2)
    """
    points_camera = torch.einsum('nij,npj->npi', rotation, points).transpose(1, 2) + translation  # (N, 3, P)

    points_img = intrinsic @ points_camera

    # Normalize the pixel coordinates
    z_coord = points_img[:, 2:3, :].clip(1e-3)
    xy = points_img[:, :2, :] / z_coord
    pixel_coords = xy.permute(0, 2, 1)  

    # Mask: valid only if in front of camera
    valid_mask = points_camera[:, 2, :] > 0

    return pixel_coords, valid_mask


def project_3d_points_to_image_batch_numpy(
        points: np.ndarray, rotation: np.ndarray, translation: np.ndarray, intrinsic: np.ndarray
    ):
    """
    Project the points using extrinsic and intrinsic to the image plane. 

    points: (P, 3) in world coordinate
    extrinsic:
      - rotation (N, 3, 3)
      - translation (N, 3, 1)
    intrinsic: (N, 3, 3)

    return: 2d points (N, P, 2)
    """
    points_camera = rotation @ points.T + translation  # (N, 3, P)

    points_img = intrinsic @ points_camera

    # Normalize the pixel coordinates
    z_coord = points_img[:, 2:3, :].clip(1e-3)
    xy = points_img[:, :2, :] / z_coord
    pixel_coords = xy.transpose(0, 2, 1)  

    # Mask: valid only if in front of camera
    valid_mask = points_camera[:, 2, :] > 0

    return pixel_coords, valid_mask


def unproject_depth_map_to_point_map_torch(
        depth_map: torch.Tensor, extrinsics: torch.Tensor, intrinsics: torch.Tensor     
):
    """
    Unproject the depth map to the world coordinate using camera parameters. 

    depth_map: (N, H, W)
    extrinsics: (N, 3, 4) world to camera
    intrinsics: (N, 3, 3)

    Return (N, H, W, 3)
    """

    world_points_list = []
    for frame_idx in range(depth_map.shape[0]):
        cur_world_points, _, _ = depth_to_world_coords_points_torch(
            depth_map[frame_idx].squeeze(-1), extrinsics[frame_idx], intrinsics[frame_idx]
        )
        world_points_list.append(cur_world_points)
    world_points_array = torch.stack(world_points_list, axis=0)

    return world_points_array


def depth_to_world_coords_points_torch(
        depth_map: torch.Tensor, extrinsic: torch.Tensor, intrinsic: torch.Tensor, eps=1e-8
):
    """
    depth_map (H, W)
    extrinsic: (3, 4)
    intrinsic: (3, 3)
    """
        
    if depth_map is None:
        return None, None, None

    # Valid depth mask
    point_mask = depth_map > eps

    # Convert depth map to camera coordinates
    cam_coords_points = depth_to_cam_coords_points_torch(depth_map, intrinsic)

    # Multiply with the inverse of extrinsic matrix to transform to world coordinates
    # extrinsic_inv is 4x4 (note closed_form_inverse_OpenCV is batched, the output is (N, 4, 4))
    cam_to_world_extrinsic = closed_form_inverse_se3(extrinsic[None])[0]

    R_cam_to_world = cam_to_world_extrinsic[:3, :3]
    t_cam_to_world = cam_to_world_extrinsic[:3, 3]

    # Apply the rotation and translation to the camera coordinates
    world_coords_points = cam_coords_points @ R_cam_to_world.T + t_cam_to_world  # HxWx3, 3x3 -> HxWx3
    # world_coords_points = np.einsum("ij,hwj->hwi", R_cam_to_world, cam_coords_points) + t_cam_to_world

    return world_coords_points, cam_coords_points, point_mask


def depth_to_cam_coords_points_torch(depth_map: torch.Tensor, intrinsic: torch.Tensor):
    """
    Convert a depth map to camera coordinates.

    depth_map: (H, W)
    intrinsic: (3, 3)
    """
    H, W = depth_map.shape
    assert intrinsic.shape == (3, 3), "Intrinsic matrix must be 3x3"
    assert intrinsic[0, 1] == 0 and intrinsic[1, 0] == 0, "Intrinsic matrix must have zero skew"

    # Intrinsic parameters
    fu, fv = intrinsic[0, 0], intrinsic[1, 1]
    cu, cv = intrinsic[0, 2], intrinsic[1, 2]

    # Generate grid of pixel coordinates
    u, v = torch.meshgrid(torch.arange(H, device=depth_map.device), torch.arange(W, device=depth_map.device))

    # Unproject to camera coordinates 
    x_cam = (u - cu) * depth_map / fu
    y_cam = (v - cv) * depth_map / fv
    z_cam = depth_map

    # Stack to form camera coordinates
    cam_coords = torch.stack((x_cam, y_cam, z_cam), axis=-1)

    return cam_coords


def closed_form_inverse_se3(se3, R=None, T=None):
    """
    Compute the inverse of each 4x4 (or 3x4) SE3 matrix in a batch.

    If `R` and `T` are provided, they must correspond to the rotation and translation
    components of `se3`. Otherwise, they will be extracted from `se3`.

    Args:
        se3: Nx4x4 or Nx3x4 array or tensor of SE3 matrices.
        R (optional): Nx3x3 array or tensor of rotation matrices.
        T (optional): Nx3x1 array or tensor of translation vectors.

    Returns:
        Inverted SE3 matrices with the same type and device as `se3`.

    Shapes:
        se3: (N, 4, 4)
        R: (N, 3, 3)
        T: (N, 3, 1)
    """
    # Check if se3 is a numpy array or a torch tensor
    is_numpy = isinstance(se3, np.ndarray)

    # Validate shapes
    if se3.shape[-2:] != (4, 4) and se3.shape[-2:] != (3, 4):
        raise ValueError(f"se3 must be of shape (N,4,4), got {se3.shape}.")

    # Extract R and T if not provided
    if R is None:
        R = se3[:, :3, :3]  # (N,3,3)
    if T is None:
        T = se3[:, :3, 3:]  # (N,3,1)

    # Transpose R
    if is_numpy:
        # Compute the transpose of the rotation for NumPy
        R_transposed = np.transpose(R, (0, 2, 1))
        # -R^T t for NumPy
        top_right = -np.matmul(R_transposed, T)
        inverted_matrix = np.tile(np.eye(4), (len(R), 1, 1))
    else:
        R_transposed = R.transpose(1, 2)  # (N,3,3)
        top_right = -torch.bmm(R_transposed, T)  # (N,3,1)
        inverted_matrix = torch.eye(4, 4)[None].repeat(len(R), 1, 1)
        inverted_matrix = inverted_matrix.to(R.dtype).to(R.device)

    inverted_matrix[:, :3, :3] = R_transposed
    inverted_matrix[:, :3, 3:] = top_right

    return inverted_matrix


def torchify(array):
    if not isinstance(array, torch.Tensor):
        return torch.from_numpy(array).to(torch.float32)

    return array

def compute_reproj(tracks, points, intrinsic, extrinsic, max_reproj=8.0):
    """
    Compute the reproj error.
    tracks: (N, P, 2)
    points: (P, 3)
    intrinsic: (N, 3, 3)
    extrinsic: (N, 3, 4)
    """
    local_points = torchify(points)
    local_tracks = torchify(tracks)
    local_extrinsic = torchify(extrinsic)
    local_trans = local_extrinsic[:, :, 3:]
    local_rot = local_extrinsic[:, :, :3]
    local_intrinsic = torchify(intrinsic)

    projected_points_2d, valid_mask = project_3d_points_to_image_batch(local_points, local_rot, local_trans, local_intrinsic)
    projected_diff = torch.norm(projected_points_2d - local_tracks, dim=-1)
    reproj_mask = projected_diff < max_reproj

    mask = (reproj_mask & valid_mask).numpy()

    return projected_diff.numpy(), mask, projected_points_2d.numpy()


def compute_depth_from_points(points, extrin):
    """
    points: (P, 3) in world coordinates
    extrin: (N, 3, 4) camera extrinsics (world -> camera)

    Returns:
        depth: (N, P)
    """
    N, _, _ = extrin.shape
    P, _ = points.shape

    # Convert to homogeneous coordinates: (P, 4)
    ones = torch.ones((P, 1), dtype=points.dtype, device=points.device)
    homog_points = torch.cat([points, ones], dim=-1)  # (P, 4)

    # Transform world -> camera for all N cameras: (N, P, 3)
    cam_points = torch.einsum('nij,pj->npi', extrin, homog_points)

    # Depth is the z-coordinate in camera space
    depth = cam_points[..., 2]  # (N, P)

    return depth
