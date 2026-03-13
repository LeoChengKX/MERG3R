import torch

from algos.utils import *
from algos.sim3utils import weighted_align_point_maps

def align_extrinsics(sequence, ba=False, device='cuda', method='umeyama'):
    if method == 'umeyama':
        transformed_results, transforms, scales = umeyama_align(sequence, ba, device=device)
    elif method == "weighted_iterative":
        transformed_results, transforms, scales = weighted_iterative_alignment(sequence, device=device)
    else:
        raise NameError("Attempted to use invalid matrix alignment method")
    
    for key in transformed_results.keys():
        if isinstance(transformed_results[key], torch.Tensor):
            transformed_results[key] = transformed_results[key].cpu().numpy() 
            if transformed_results[key].shape[0] == 1:
                transformed_results[key] = transformed_results[key].squeeze(0) # remove batch dimension

    return transformed_results, transforms, scales


def compute_similarity(tgt_points, src_points):
    A = tgt_points
    B = src_points

    mA = np.mean(A, axis=1)[:, None]
    mB = np.mean(B, axis=1)[:, None]

    varB = np.mean(np.linalg.norm((B - mB), ord=2, axis=0) ** 2)


    covar = ((A - mA) @ (B - mB).T) / A.shape[1]
    U, D, Vt = np.linalg.svd(covar, full_matrices=True)

    covar_rank = np.linalg.matrix_rank(covar)
    
    S = np.eye(3)
    if (covar_rank >= 2) and varB > 0:
        if (np.linalg.det(U) * np.linalg.det(Vt.T)) < 0: # Eqn 39
            S[-1, -1] = -1
        c = (1/varB) * np.trace(np.diag(D) @ S) # Eqn 42
        R = (U @ S @ Vt) # Eqn 40
    else:
        return None
    
    t = mA - (c * (R @ mB)) # Eqn 41
    return R, t, c

def ransac_umeyama(tgt_points, src_points, min_samples=4, inlier_dist=1e-03, iterations=500):
    best_inliers = np.array([])
    N = src_points.shape[1] # Num points
    best_iter = 0
    for i in range(iterations):
        sample_idx = np.random.choice(N, min_samples, replace=False)
        src_samples = src_points[:, sample_idx] # 3 x samples
        tgt_samples = tgt_points[:, sample_idx] # 3 x samples
        sim = compute_similarity(tgt_samples, src_samples)
        if sim is None: continue
        R, t, c = sim
        transformed_src = (c * (R @ src_points)) + t # 3 x N
        diffs = np.linalg.norm(tgt_points - transformed_src, axis=0)
        inliers = diffs <= inlier_dist
        if np.sum(inliers) > np.sum(best_inliers):
            best_inliers = inliers
            best_iter = i
    
    if np.sum(best_inliers) >= min_samples:
        print("[ALIGNMENT] Best RANSAC Iter: ", best_iter)
        sim = compute_similarity(tgt_points[:, best_inliers], src_points[:, best_inliers])
        return sim
    return None



def valid_sim(R, t, c):
    if R is None: return False
    if not (np.isfinite(R).all() and np.isfinite(t).all() and np.isfinite(c)): return False
    if not np.allclose(R.T @ R, np.eye(3), atol=1e-3): return False
    if not (0.01 <= c <= 100.0): return False
    return True

def robust_align(tgt_points, src_points, inlier_dist=1e-03):
    # 1) RANSAC
    sim = ransac_umeyama(tgt_points, src_points, inlier_dist=inlier_dist, iterations=500)
    if sim and valid_sim(*sim): return sim

    # 2) LOOSENED RANSAC
    print("[ALIGNMENT] Failed first RANSAC alignment attempt. Loosening inlier threshold")
    for tau2 in (inlier_dist*5, inlier_dist*10):
        sim = ransac_umeyama(tgt_points, src_points, inlier_dist=tau2, iterations=1500)
        if sim and valid_sim(*sim): return sim

    # 3) NO THRESHOLD FALLBACK (only if it passes validation)
    print("[ALIGNMENT] Failed second RANSAC alignment attempt. Attempting to align all points")
    sim = compute_similarity(tgt_points, src_points)
    if sim and valid_sim(*sim): return sim
    return None

def umeyama_align(sequence, ba, device="cuda"):
    # --------------------------------------------------------------
    # Compute transformation matrices
    transforms = {} # Matrices transforming child frames into parent frames
    scales = {}
    subsample = 1
    for parent_id in sequence.edges:
        parent = sequence.edges[parent_id]
        for child_id in parent:
            edge = parent[child_id]
            transform = torch.zeros((3,4)).to(device)

            parent_overlap_indices, child_overlap_indices = edge['parent_overlap_indices'], edge['child_overlap_indices']
            prev_set = sequence.predictions[parent_id]
            curr_set = sequence.predictions[child_id]
            A_depths, A_intri, A_extri = prev_set['depth'][parent_overlap_indices], prev_set['intrinsic'][parent_overlap_indices], prev_set['extrinsic'][parent_overlap_indices]
            B_depths, B_intri, B_extri = curr_set['depth'][child_overlap_indices], curr_set['intrinsic'][child_overlap_indices], curr_set['extrinsic'][child_overlap_indices]
            A_depth_conf, B_depth_conf = prev_set['depth_conf'][parent_overlap_indices, ..., None], curr_set['depth_conf'][child_overlap_indices, ..., None]
            # Use percentiles instead of hard thresholds
            percentile = 0.7
            depth_thresh_A = np.min(A_depth_conf) + (percentile * (np.max(A_depth_conf) - np.min(A_depth_conf)))
            depth_thresh_B = np.min(B_depth_conf) + (percentile * (np.max(B_depth_conf) - np.min(B_depth_conf)))
            depth_conf = min(depth_thresh_A, depth_thresh_B)
            depth_mask = (A_depth_conf > depth_conf) & (B_depth_conf > depth_conf)

            # If no BA then just use regular world points rather than bundle adjusted points
            if ba: # Use bundle adjusted points for scale and extrinsics for translation + rotation
                _, H, W , _= prev_set['depth'].shape
                S = len(prev_set['tracks'][parent_overlap_indices])
                A_tracks, A_points = (np.rint(prev_set['tracks'][parent_overlap_indices])).astype(np.int32), prev_set['points']
                B_tracks, B_points =  (np.rint(curr_set['tracks'][child_overlap_indices])).astype(np.int32), curr_set['points']


                # # compute span including inclusive endpoints
                row_max = max(np.max(A_tracks[...,1]), np.max(B_tracks[...,1]), H - 1)
                row_min = min(np.min(A_tracks[...,1]), np.min(B_tracks[...,1]), 0)
                col_max = max(np.max(A_tracks[...,0]), np.max(B_tracks[...,0]), W - 1)
                col_min = min(np.min(A_tracks[...,0]), np.min(B_tracks[...,0]), 0)

                rows = row_max - row_min + 1
                cols = col_max - col_min + 1
                
                A_image_to_3D = np.full((S, rows, cols, 3), np.nan, dtype=np.float32)
                B_image_to_3D = np.full((S, rows, cols, 3), np.nan, dtype=np.float32)

                row_offset = -row_min
                col_offset = -col_min

                # Add row and col offsets to center true image in padded image
                A_image_to_3D[np.arange(S)[:, None], A_tracks[...,1] + row_offset, A_tracks[...,0] + col_offset] = A_points
                B_image_to_3D[np.arange(S)[:, None], B_tracks[...,1] + row_offset, B_tracks[...,0] + col_offset] = B_points

                # Crop 
                A_image_to_3D = A_image_to_3D[: , row_offset : row_offset + H, col_offset : col_offset + W]
                B_image_to_3D = B_image_to_3D[: , row_offset : row_offset + H, col_offset : col_offset + W]

                assert (np.all(A_tracks[...,1]) + row_offset >= 0) and (np.all(A_tracks[...,1]) + row_offset < rows)
                assert (np.all(B_tracks[...,1]) + row_offset >= 0) and (np.all(B_tracks[...,1]) + row_offset < rows)
                assert (np.all(A_tracks[...,0]) + col_offset >= 0) and (np.all(A_tracks[...,0]) + col_offset < cols)
                assert (np.all(B_tracks[...,0]) + col_offset >= 0) and (np.all(B_tracks[...,0]) + col_offset < cols)

                # Validity Filter
                A_valid = ~np.isnan(A_image_to_3D).any(axis=-1)
                B_valid = ~np.isnan(B_image_to_3D).any(axis=-1) 
                valid_mask = A_valid & B_valid # [B, H, W]

                # # Create lists A and B of shape [N, 3] that represent overlapping bundle adjusted points between images
                A = A_image_to_3D[valid_mask] # [N, 3]
                B = B_image_to_3D[valid_mask] # [N, 3]


                A = A.transpose(1,0) # 3 x N
                B = B.transpose(1, 0) # 3 x N
                
                R, t, c = robust_align(A, B)
                
            else:
                A = unproject_depth_map_to_point_map(A_depths, A_extri, A_intri) # Target frame (y)
                A = A[depth_mask.squeeze(-1)]
                A = A[::subsample, :]

                B = unproject_depth_map_to_point_map(B_depths, B_extri, B_intri) # Source frame (x)
                B = B[depth_mask.squeeze(-1)]
                B = B[::subsample, :]


                A = A.transpose(1,0) # 3 x N
                B = B.transpose(1, 0) # 3 x N

                sim = compute_similarity(A, B)
                assert sim is not None, "[ALIGNMENT] Failed to find a good similarity transform"
                R, t, c = sim
                

            # Sanity checks
            # MSE between transformed points
            umeyama_mse = np.mean(np.linalg.norm(A - ((c * (R @ B)) + t), ord=2, axis=0) ** 2)
            print("[ALIGNMENT] UMEYAMA MSE: ", umeyama_mse)

            print("[ALIGNMENT] SCALE: ", c)

            transform[:3, :3] = torch.from_numpy(R).float().to(device)
            transform[:3, 3] = torch.from_numpy(t.squeeze(-1)).float().to(device)

            if parent_id not in transforms: transforms[parent_id] = {}
            if parent_id not in scales: scales[parent_id] = {}
            # Transform: child -> parent
            transforms[parent_id][child_id] = transform
            scales[parent_id][child_id] = c
    
    # --------------------------------------------------------------
    # Transform all poses to shared coordinate frame

    return sequence.transform_to_shared_frame(transforms, scales), transforms, scales


def align_inv_depth_to_depth(
    source_depth: torch.Tensor,
    target_depth: torch.Tensor,
    target_mask: torch.Tensor | None = None,
    quantile_masking: bool = True,
) -> tuple[torch.Tensor, float, float]:
    """
    Apply affine transformation to align source inverse depth to target depth.

    Args:
        source_inv_depth: Inverse depth map to be aligned. Shape: (H, W).
        target_depth: Target depth map. Shape: (H, W).
        target_mask: Mask of valid target pixels. Shape: (H, W).

    Returns:
        Aligned Depth map. Shape: (H, W).
        scale: Scaling factor.
        bias: Bias term.
    """
    source_inv_depth = 1.0 / source_depth
    target_inv_depth = 1.0 / target_depth
    source_mask = source_inv_depth > 0
    target_depth_mask = target_depth > 0

    if target_mask is None:
        target_mask = target_depth_mask
    else:
        target_mask = torch.logical_and(target_mask > 0, target_depth_mask)

    # Remove outliers
    if quantile_masking:
        outlier_quantiles = torch.tensor([0.1, 0.9], device=source_inv_depth.device)
        source_data_low, source_data_high = torch.quantile(source_inv_depth[source_mask], outlier_quantiles)
        target_data_low, target_data_high = torch.quantile(target_inv_depth[target_mask], outlier_quantiles)
        source_mask = (source_inv_depth > source_data_low) & (source_inv_depth < source_data_high)
        target_mask = (target_inv_depth > target_data_low) & (target_inv_depth < target_data_high)

    mask = torch.logical_and(source_mask, target_mask)

    source_data = source_inv_depth[mask].view(-1, 1)
    target_data = target_inv_depth[mask].view(-1, 1)

    n = source_data.shape[0]
    # CUDA lstsq requires overdetermined system (rows >= cols); need at least 2 samples for 2 unknowns
    if n < 2:
        scale = torch.tensor(1.0, device=source_inv_depth.device)
        bias = torch.tensor(0.0, device=source_inv_depth.device)
        aligned_inv_depth = source_inv_depth * scale + bias
        aligned_depth = torch.clamp(aligned_inv_depth.reciprocal(), min=1e-4)
        return aligned_depth, scale.item(), bias.item()

    ones = torch.ones((n, 1), device=source_data.device)
    source_data_h = torch.cat([source_data, ones], dim=1)
    transform_matrix = torch.linalg.lstsq(source_data_h, target_data).solution

    scale = transform_matrix[0, 0]
    bias = transform_matrix[1, 0]
    aligned_inv_depth = source_inv_depth * scale + bias
    aligned_depth = torch.clamp(aligned_inv_depth.reciprocal(), min=1e-4)

    return aligned_depth, scale.item(), bias.item()

def weighted_iterative_alignment(sequence, device="cuda"):
    transforms = {} # Matrices transforming child frames into parent frames
    scales = {}
    for parent_id in sequence.edges:
        parent = sequence.edges[parent_id]
        for child_id in parent:
            edge = parent[child_id]
            transform = torch.zeros((3,4)).to(device)

            parent_overlap_indices, child_overlap_indices = edge['parent_overlap_indices'], edge['child_overlap_indices']
            prev_set = sequence.predictions[parent_id]
            curr_set = sequence.predictions[child_id]
            A_depths, A_intri, A_extri = prev_set['depth'][parent_overlap_indices], prev_set['intrinsic'][parent_overlap_indices], prev_set['extrinsic'][parent_overlap_indices]
            B_depths, B_intri, B_extri = curr_set['depth'][child_overlap_indices], curr_set['intrinsic'][child_overlap_indices], curr_set['extrinsic'][child_overlap_indices]
            A_depth_conf, B_depth_conf = prev_set['depth_conf'][parent_overlap_indices, ..., None], curr_set['depth_conf'][child_overlap_indices, ..., None]
            


            # Proper 0–1 normalization
            normalized_depth_confs_A = (A_depth_conf - np.min(A_depth_conf)) / (np.max(A_depth_conf) - np.min(A_depth_conf))
            normalized_depth_confs_B = (B_depth_conf - np.min(B_depth_conf)) / (np.max(B_depth_conf) - np.min(B_depth_conf))

            percentile = 70
            conf_threshold = min(np.percentile(normalized_depth_confs_A, percentile), np.percentile(normalized_depth_confs_B, percentile))

            A = unproject_depth_map_to_point_map(A_depths, A_extri, A_intri) # Target frame (y)
            B = unproject_depth_map_to_point_map(B_depths, B_extri, B_intri) # Source frame (x)

            c, R, t = weighted_align_point_maps(A, 
                                    normalized_depth_confs_A.squeeze(), 
                                    B, 
                                    normalized_depth_confs_B.squeeze(), 
                                    conf_threshold=conf_threshold,
                                    align_method="numpy")


            transform[:3, :3] = torch.from_numpy(R).float().to(device)
            transform[:3, 3] = torch.from_numpy(t).float().to(device)

            if parent_id not in transforms: transforms[parent_id] = {}
            if parent_id not in scales: scales[parent_id] = {}
            # Transform: child -> parent
            transforms[parent_id][child_id] = transform
            scales[parent_id][child_id] = c

    return sequence.transform_to_shared_frame(transforms, scales), transforms, scales
