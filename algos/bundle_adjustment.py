import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
import roma

from algos.utils import *
from algos.geometry import  project_3d_points_to_image_batch, project_3d_points_to_image_torch
from algos.tracking import *


def reprojection_loss(points_pixel, tracks, valid_mask, depth_conf_points=None, delta=2.0, eps=1e-12):
    """
    points_pixel, tracks: (N, P, 2) in pixels
    valid_mask:          (N, P) bool or 0/1
    depth_conf_points:   (P,) optional per-track weight (e.g., depth/confidence)
    delta:               Huber delta (pixels)
    """
    # per-observation robust residual (N,P,2) -> (N,P)
    per_comp = F.smooth_l1_loss(points_pixel, tracks, beta=delta, reduction="none")  # (N,P,2)
    per_obs  = per_comp.mean(dim=-1)  # (N,P)

    # weights
    if depth_conf_points is None:
        w_p = torch.ones((per_obs.shape[1],), device=per_obs.device, dtype=per_obs.dtype)
    else:
        w_p = depth_conf_points.to(per_obs.dtype).to(per_obs.device).clamp_min(eps)  # (P,)

    w = w_p                   # (N,P)
    if valid_mask is not None:
        w = w * valid_mask.to(per_obs.dtype)                    # zero out invalid obs

    # weighted MEAN over valid obs
    denom = w.sum().clamp_min(eps)
    loss = (w * per_obs).sum() / denom
    return loss


def gradient_bundle_adjustment(points: np.ndarray, 
                               extrinsic: np.ndarray, 
                               intrinsic: np.ndarray, 
                               tracks: list[np.ndarray], 
                               points_id: list[np.ndarray],
                               depth_conf_points: np.ndarray,
                               depth_map: np.ndarray,
                               depth_conf: np.ndarray,
                               shared_intrinsics=False,
                               optimize_points=True,
                               optimize_intrinsics=False,
                               # --------------------------
                               epoch=10,
                               lr=1e-4,
                               max_reproj_error=None,
                               device="cuda"):
    """
    points: (P, 3)
    extrinsic: (N, 3, 4), w2c
    intrinsic: (N, 3, 3)
    tracks: (N, P, 2)
    track_mask: (N, P)
    depth_conf_points: (P)
    depth_map: (N, H, W, 1)
    depth_conf: (N, H, W)
    """
    N, H, W, _ = depth_map.shape
    P = points.shape[0]
    
    extrinsic = torch.from_numpy(extrinsic).to(device)
    intrinsic = torch.from_numpy(intrinsic).to(device)
    points = torch.from_numpy(points).to(torch.float32).to(device)
    tracks = [torch.from_numpy(track).to(device) for track in tracks]
    points_id = [torch.from_numpy(idx).to(device) for idx in points_id]
    depth_conf_points = torch.from_numpy(depth_conf_points).to(device)
    depth_map = torch.from_numpy(depth_map).to(device)
    depth_conf = torch.from_numpy(depth_conf).to(device)

    num_images = extrinsic.shape[0]

    def make_intri_extri(log_focals, pps, quats, trans):
        all_rotation = roma.unitquat_to_rotmat(quats)
       
        w2c_poses = torch.zeros((all_rotation.shape[0], 4, 4), device=device)
        w2c_poses[:, :3, :3] = all_rotation
        w2c_poses[:, :3, 3:] = trans


        if shared_intrinsics:
            f_batch = torch.exp(log_focals[0]).repeat(N)
            pp_batch = pps[0].repeat(N, 1)
        else:
            f_batch = torch.exp(torch.stack(log_focals))
            pp_batch = torch.stack(pps)

        K = torch.zeros((N, 3, 3), device=device)
        K[:, 0, 0] = f_batch
        K[:, 1, 1] = f_batch
        K[:, 0, 2] = pp_batch[:, 0]
        K[:, 1, 2] = pp_batch[:, 1]
        K[:, 2, 2] = 1.0
    
        return K, w2c_poses

    # Optimization paramters
    # Extrinsics, initialize with relative pose
    quats = roma.rotmat_to_unitquat(extrinsic[:, :3, :3])
    trans = extrinsic[:, :3, 3:]


    quats = nn.Parameter(quats, requires_grad=True)
    trans = nn.Parameter(trans, requires_grad=True)
    # 3D points
    points = nn.Parameter(points, requires_grad=False)
    # Intrinsics
    focals_x = intrinsic[:, 0, 0]
    focals_y = intrinsic[:, 1, 1]
    log_focals = torch.log((focals_x + focals_y) / 2)
    pps = intrinsic[:, :2, 2]

    if shared_intrinsics:
        log_focals = log_focals.mean()
        log_focals_param = nn.Parameter(log_focals, requires_grad=optimize_intrinsics)
        log_focals = [log_focals_param for _ in range(num_images)]
        pps = pps.view(-1, 2).mean(dim=0)
        pps_param = nn.Parameter(pps, requires_grad=optimize_intrinsics)
        pps = [pps_param for _ in range(num_images)]
    else:
        log_focals = [nn.Parameter(log_focals[i]) for i in range(num_images)]
        pps = [nn.Parameter(pps[i]) for i in range(num_images)]

    track_mask = []
    if max_reproj_error is not None:
        # Pre-filter points with high reproj error as outliers
        for img in range(N):
            img_points_id = points_id[img]
            if len(img_points_id) == 0:
                track_mask.append(None)
                continue
            img_points = points[img_points_id]
            projected_points_2d, valid_mask = project_3d_points_to_image_torch(img_points, extrinsic[img, :, :3], extrinsic[img, :, 3:], intrinsic[img])
            projected_diff = torch.norm(projected_points_2d - tracks[img], dim=-1)
            
            reproj_mask = projected_diff < max_reproj_error
            reproj_mask = reproj_mask & valid_mask
            
            track_mask.append(reproj_mask)


    # Start BA
    optimize_params = [quats, trans]
    if optimize_points:
        optimize_params.append(points)
        points.requires_grad_(True)
    if optimize_intrinsics:
        optimize_params.extend(log_focals)
        optimize_params.extend(pps)

    optimizer = Adam(optimize_params, lr=lr, weight_decay=0)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epoch, eta_min=1e-4
    )

    optimize_depth_conf_points = []
    max_num_points = 0

    for img in range(N):
        if len(points_id[img]) == 0: 
            optimize_depth_conf_points.append(None)
            continue
        optimize_depth_conf_points.append(depth_conf_points[points_id[img]])

        if len(points[points_id[img]]) > max_num_points:
            max_num_points = len(points[points_id[img]])
    
    # Extend to common length by padding
    for img in range(N):
        if optimize_depth_conf_points[img] is None: 
            optimize_depth_conf_points[img] = torch.zeros(max_num_points,device=device)
            tracks[img] = torch.zeros((max_num_points, 2), device=device) 
            track_mask[img] = torch.zeros(max_num_points, device=device) 
        else:
            depth_conf = optimize_depth_conf_points[img]    # (P_i, 1)
            depth_conf = torch.cat((depth_conf, torch.zeros(max_num_points - depth_conf.shape[0], device=device)))  
            optimize_depth_conf_points[img] = depth_conf

            img_track = tracks[img]   # (P_i, 2)
            img_track = torch.cat((img_track, torch.zeros(max_num_points - img_track.shape[0], 2, device=device))) 
            tracks[img] = img_track

            mask = track_mask[img]   # (P_i)
            mask = torch.cat((mask, torch.zeros(max_num_points - mask.shape[0], device=device))) 
            track_mask[img] = mask

    # optimize_point = torch.stack(optimize_point)
    optimize_depth_conf_points = torch.stack(optimize_depth_conf_points)
    tracks = torch.stack(tracks)
    track_mask = torch.stack(track_mask).to(torch.bool)
    
    with tqdm(total=epoch) as bar:
        for _ in range(epoch):
            # Construct the loss function
            loss = torch.zeros(1, device=device)

            K, w2c = make_intri_extri(log_focals, pps, quats, trans)

            optimize_point = []
            for img in range(N):
                if len(points_id[img]) == 0:
                    optimize_point.append(torch.zeros((max_num_points, 3), device=device, requires_grad=True))
                    continue
                img_points = points[points_id[img]]
                optimize_point.append(torch.cat((img_points, torch.zeros(max_num_points - img_points.shape[0], 3, device=device))))

            optimize_point = torch.stack(optimize_point)

            points_pixel, mask = project_3d_points_to_image_batch(optimize_point, w2c[:, :3, :3], w2c[:, :3, 3:], K)  # (P, 2)
            valid_mask = track_mask & mask

            loss = reprojection_loss(points_pixel, tracks, valid_mask, optimize_depth_conf_points)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            # Normalize quaternion to unit
            with torch.no_grad():
                quats.data = F.normalize(quats.data, p=2, dim=1)

            bar.set_postfix_str(f'loss={loss.item():.3f},lr={scheduler.get_last_lr()[0]:.5f}')
            bar.update(1)
    
    print(f"Reprojection final loss: {loss.item():.3f}. ")
    
    extrinsic = w2c
    intrinsic = K

    return extrinsic.detach().cpu().numpy(), intrinsic.detach().cpu().numpy(), \
                points.detach().cpu().numpy(), track_mask


def global_bundle_adjustment(prediction, track, points_id, points_3d, points_conf, max_reproj_error=8.0, shared_camera=True, lr=1e-4, epoch=300):

    # Filter points by confidence
    conf_thres_value = np.percentile(points_conf, 30.0)
    filtered_flag = points_conf > conf_thres_value


    extrinsic = prediction['extrinsic']
    intrinsic = prediction['intrinsic']
    depth_map = prediction['depth']
    depth_conf = prediction['depth_conf']

    extrinsic, intrinsic, points3D, valid_track_mask = gradient_bundle_adjustment(
            points_3d,
            extrinsic,
            intrinsic,
            track,
            points_id,
            points_conf,
            depth_map,
            depth_conf,
            optimize_intrinsics=False,
            optimize_points=True,
            shared_intrinsics=shared_camera,
            epoch=epoch,
            lr=lr,
            max_reproj_error=max_reproj_error
        )

    prediction['extrinsic'] = extrinsic
    prediction['intrinsic'] = intrinsic
    prediction['points'] = points3D

    return filtered_flag, valid_track_mask

