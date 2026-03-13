import torch
import torch.nn.functional as F
import numpy as np
from scipy.spatial import KDTree
from copy import deepcopy

from vggt.models.vggt import VGGT
from vggt.dependency.vggsfm_utils import initialize_feature_extractors, extract_keypoints, farthest_point_sampling
from algos.sequence import Sequence


def select_query_points_all_images(images, extractor_method="aliked+sp"):
    query_points = []
    extractors = initialize_feature_extractors(max_query_num=1024, extractor_method=extractor_method)

    for image in images:
        query_points.append(extract_keypoints(image, extractors, round_keypoints=False))

    return query_points


def generate_similarity_matrix(frame_feat, spatial_similarity=False):
     # Process features based on similarity type
    if spatial_similarity:
        frame_feat = frame_feat["x_norm_patchtokens"]
        frame_feat_norm = F.normalize(frame_feat, p=2, dim=1)

        # Compute the similarity matrix
        frame_feat_norm = frame_feat_norm.permute(1, 0, 2)
        similarity_matrix = torch.bmm(frame_feat_norm, frame_feat_norm.transpose(-1, -2))
        similarity_matrix = similarity_matrix.mean(dim=0)
    else:
        frame_feat = frame_feat["x_norm_clstoken"]
        frame_feat_norm = F.normalize(frame_feat, p=2, dim=1)
        similarity_matrix = torch.mm(frame_feat_norm, frame_feat_norm.transpose(-1, -2))

    distance_matrix = 100 - similarity_matrix.clone()

    
    return similarity_matrix, distance_matrix



def select_query_frames_and_track(sequence: Sequence, model: VGGT, all_query_points,
                                  node_id, parent_id=None, query_frames=None,
                                  vis_threshold=0.7, max_query_points=1024):
    """
    Recursive function for BFS on the MST of sequence to select the query frames for each
    cluster using the edge connetivity. After that, perform a tracking using CoTracker. 
    """
    if query_frames is None:
        query_frames = {}
        
    predictions = sequence.predictions
    # TODO: Better to change this to ordered dict
    edges = sequence.edges
    curr_node = predictions[node_id]
    subset_to_img_ids = sequence.subset_to_img_ids

    # True when the current node is the root node
    is_top_root = (parent_id is None) or (parent_id == node_id)
    # True when current node has no children in the tree
    is_leaf = node_id not in edges

    # Only one frame
    if is_leaf and is_top_root:
        # TODO: Add DINO selection logic
        return None

    if not is_leaf:
        child_ids = edges[node_id]
        for idx in child_ids:
            child_node = predictions[idx]
            child_overlap_indices = edges[node_id][idx]['child_overlap_indices']
            parent_overlap_indices = edges[node_id][idx]['parent_overlap_indices']
            # TODO: Smarter selection heuristic e.g. merge same indices or fps between the overlapping images # TODO:
            # indices_idx = np.random.choice(len(child_overlap_indices))
            similarity_matrix_parent, _ = generate_similarity_matrix(curr_node['dino_features'], spatial_similarity=False)
            similarity_matrix_child, _ = generate_similarity_matrix(child_node['dino_features'], spatial_similarity=False)
            # Ignore self-pairing
            similarity_matrix_parent.fill_diagonal_(0)
            similarity_mean_parent = similarity_matrix_parent.sum(dim=1) / (similarity_matrix_parent.shape[0] - 1)
            similarity_matrix_child.fill_diagonal_(0)
            similarity_mean_child = similarity_matrix_child.sum(dim=1) / (similarity_matrix_child.shape[0] - 1)

            overlap_similarity_mean_parent = similarity_mean_parent[parent_overlap_indices]
            overlap_similarity_mean_child = similarity_mean_child[child_overlap_indices]
            overlap_similarity_matrix_parent = similarity_matrix_parent[parent_overlap_indices]
            
    
            if is_top_root:
                # Find the most common frame in both prent and child
                indices_idx = torch.argmax(overlap_similarity_mean_parent).item()
            
            else:
                # Select the frame with the most distance from parent
                parent_query = query_frames[node_id]["from_parent"]
                # Maximize similarity within the overlapping images
                scores = overlap_similarity_mean_parent + \
                    overlap_similarity_mean_child - similarity_matrix_parent[parent_query][parent_overlap_indices]
                indices_idx = torch.argmax(scores)

            child_img_id = child_overlap_indices[indices_idx]
            parent_img_id = parent_overlap_indices[indices_idx]
            if node_id in query_frames.keys():
                query_frames[node_id]["from_child"].append(parent_img_id)
            else:
                query_frames[node_id] = {"from_child": [parent_img_id], "from_parent": None}
            
            # Create child info
            query_frames[idx] = {"from_child": [], "from_parent": child_img_id}

        for idx in child_ids:
            select_query_frames_and_track(sequence, model, all_query_points, idx, parent_id=node_id, query_frames=query_frames)
    
    # Start tracking, 0 is always the parent image
    if is_leaf:
        # For leaf node there is no child and can only query the frames from parent
        image_from_parent = query_frames[node_id]['from_parent']
        image_from_parent_global = subset_to_img_ids[node_id][image_from_parent]

        query_frame_ids = generate_rank_from_dino_features(curr_node['dino_features'], 3, first_frame_ids=image_from_parent)
        curr_query_points = [all_query_points[image_from_parent_global], 
                             all_query_points[subset_to_img_ids[node_id][query_frame_ids[1]]],
                             all_query_points[subset_to_img_ids[node_id][query_frame_ids[2]]]]

        print(f"Predicting tracks for node {node_id}...")
        print(f"Num of query frames: {len(query_frame_ids)}")
        (
            tracks,
            vis_scores,
            conf_scores,
            world_points,
            world_points_conf
        ) =  predict_image_tracks(
                                model,
                                curr_query_points,
                                curr_node['features'],
                                points_3d=curr_node['world_points'],
                                depth_conf=curr_node['depth_conf'],
                                query_frame_indexes=query_frame_ids
                                )
        

    else:
        # For non-leaf node use the query points tracked from the the child cluster
        query_frame_ids = query_frames[node_id]['from_child']

        child_ids = edges[node_id]
        curr_query_points = []
        # Order may not be the same since it is a dict!
        for idx in child_ids:
            overlap_child_idx = query_frames[idx]['from_parent']
            child_track = predictions[idx]['tracks'][overlap_child_idx]  # (P, 2)
            child_vis = predictions[idx]['vis_scores'][overlap_child_idx]  # (P)

            valid_mask = child_vis > vis_threshold
            # query_points = child_track[valid_mask]

            # TODO: Maybe add another if for query points less then the number, i.e. for too few queries
            # if query_points.shape[0] > max_query_points:
            #     # Subsample the query points to be under the threshold
            #     indices = np.random.choice(len(query_points), size=max_query_points, replace=False)
            #     query_points = query_points[indices]
            query_points = all_query_points[subset_to_img_ids[idx][overlap_child_idx]]
            
            curr_query_points.append(query_points)
        
        if query_frames[node_id]['from_parent'] is not None:
            parent_image_id = query_frames[node_id]['from_parent']
            if parent_image_id in query_frame_ids:
                query_frame_ids.remove(parent_image_id)
            
            parent_in_global_idx = subset_to_img_ids[node_id][parent_image_id]
            query_frame_ids = [parent_image_id, *query_frame_ids]
            curr_query_points = [all_query_points[parent_in_global_idx], *curr_query_points]
        
        if len(query_frame_ids) == 1:
            image_idx_global = subset_to_img_ids[node_id][query_frame_ids[0]]

            query_frame_ids = generate_rank_from_dino_features(curr_node['dino_features'], 3, first_frame_ids=query_frame_ids[0])
            curr_query_points = [all_query_points[image_idx_global], 
                                 all_query_points[subset_to_img_ids[node_id][query_frame_ids[1]]],
                                 all_query_points[subset_to_img_ids[node_id][query_frame_ids[2]]]
                                 ]
        elif len(query_frame_ids) == 2:
            pass
            image_idx_global = subset_to_img_ids[node_id][query_frame_ids[0]]
            print(query_frame_ids)
            # TODO: Change this
            query_frame_ids = generate_rank_from_dino_features(curr_node['dino_features'], 3, first_frame_ids=query_frame_ids[0])
            curr_query_points = [all_query_points[image_idx_global], 
                                 all_query_points[subset_to_img_ids[node_id][query_frame_ids[1]]],
                                 all_query_points[subset_to_img_ids[node_id][query_frame_ids[2]]]
                                 ]

        print(f"Predicting tracks for node {node_id}...")
        print(f"Num of query frames: {len(query_frame_ids)}")
        (
            tracks,
            vis_scores,
            conf_scores,
            world_points,
            world_points_conf
        ) =  predict_image_tracks(
                                model,
                                curr_query_points,
                                curr_node['features'],
                                points_3d=curr_node['world_points'],
                                depth_conf=curr_node['depth_conf'],
                                query_frame_indexes=query_frame_ids
                                )

    curr_node['tracks'] = tracks
    curr_node['vis_scores'] = vis_scores
    curr_node['points'] = world_points
    curr_node['points_conf'] = world_points_conf
    curr_node['tracker_conf'] = conf_scores

    # Delete the dino features since we don't need them for the downstream
    features = curr_node.pop("dino_features")
    del features
    
    return query_frames


def generate_rank_by_dino(
    images, query_frame_num, dino_model, image_size=336, device="cuda", spatial_similarity=False
):
    """
    Generate a ranking of frames using DINO ViT features.

    Args:
        images: Tensor of shape (S, 3, H, W) with values in range [0, 1]
        query_frame_num: Number of frames to select
        image_size: Size to resize images to before processing
        model_name: Name of the DINO model to use
        device: Device to run the model on
        spatial_similarity: Whether to use spatial token similarity or CLS token similarity

    Returns:
        List of frame indices ranked by their representativeness
    """
    # Resize images to the target size
    images = F.interpolate(images, (image_size, image_size), mode="bilinear", align_corners=False)

    # Normalize images using ResNet normalization
    _RESNET_MEAN = [0.485, 0.456, 0.406]
    _RESNET_STD = [0.229, 0.224, 0.225]
    resnet_mean = torch.tensor(_RESNET_MEAN, device=device).view(1, 3, 1, 1)
    resnet_std = torch.tensor(_RESNET_STD, device=device).view(1, 3, 1, 1)
    images_resnet_norm = (images - resnet_mean) / resnet_std

    with torch.no_grad():
        frame_feat = dino_model(images_resnet_norm, is_training=True)

    similarity_matrix, distance_matrix = generate_similarity_matrix(frame_feat, spatial_similarity=spatial_similarity)

    # Ignore self-pairing
    similarity_matrix.fill_diagonal_(-100)
    similarity_sum = similarity_matrix.sum(dim=1)

    # Find the most common frame
    most_common_frame_index = torch.argmax(similarity_sum).item()

    # Conduct FPS sampling starting from the most common frame
    fps_idx = farthest_point_sampling(distance_matrix, query_frame_num, most_common_frame_index)

    # Clean up all tensors and models to free memory
    del frame_feat, frame_feat_norm, similarity_matrix, distance_matrix
    # del dino_model
    torch.cuda.empty_cache()

    return fps_idx


def generate_rank_from_dino_features(frame_feat, query_frame_num, spatial_similarity=False, first_frame_ids=[]):
    similarity_matrix, distance_matrix = generate_similarity_matrix(frame_feat, spatial_similarity=spatial_similarity)

    # Ignore self-pairing
    similarity_matrix.fill_diagonal_(-100)
    similarity_sum = similarity_matrix.sum(dim=1)

    if isinstance(first_frame_ids, int):
        first_frame_ids = first_frame_ids
    elif len(first_frame_ids) == 0: 
        # Find the most common frame
        first_frame_idx = torch.argmax(similarity_sum).item()
        first_frame_ids = first_frame_idx
    

    # Conduct FPS sampling starting from the most common frame
    fps_idx = farthest_point_sampling(distance_matrix, query_frame_num, first_frame_ids)

    # Clean up all tensors to clean up memory
    del similarity_matrix, distance_matrix

    return fps_idx


def predict_track_from_cotracker(model: VGGT, feature_maps, query_points, frame_idx, iters=4):
    """
    Predict a trcak using VGGT tracking head.

    model: VGGT
    feature_maps: (N, H, W, F)
    query_points: (N, 2)
    frame_idx: int
    """

    # Make the frame_idx image the first for tracking
    # Reorder images to put query image first
    # fm = feature_maps.clone()
    fm = torch.from_numpy(feature_maps)
    fm[[0, frame_idx]] = fm[[frame_idx, 0]]
    feature_maps = fm.unsqueeze(0).to("cuda")
    
    # Use CoTracker
    with torch.no_grad():
        # with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        coord_preds, vis_scores, conf_scores = model.track_head.tracker(query_points=query_points, fmaps=feature_maps, iters=iters)
    
    # Only the last track is needed
    coord_preds = coord_preds[-1]

    # Restore original order
    coord_preds, vis_scores, conf_scores = coord_preds.squeeze(0), vis_scores.squeeze(0), conf_scores.squeeze(0)
    coord_preds[[0, frame_idx]] = coord_preds[[frame_idx, 0]]
    vis_scores[[0, frame_idx]] = vis_scores[[frame_idx, 0]]
    conf_scores[[0, frame_idx]] = conf_scores[[frame_idx, 0]]
    coord_preds, vis_scores, conf_scores = coord_preds[None], vis_scores[None], conf_scores[None]
    
    return coord_preds, vis_scores, conf_scores


def predict_image_tracks(model: VGGT, query_frame_keypoints, feature_maps, query_frame_indexes, 
                         points_3d, depth_conf):
    
    """
    query_frame should correspond to number of query_frame_indexes. 
    """
    pred_tracks = []
    pred_vis_scores = []
    pred_confs = []
    pred_points_3d = []
    pred_points_conf = []

    for i, frame_idx in enumerate(query_frame_indexes):
        tracks, vis, conf = predict_track_from_cotracker(model, feature_maps, query_frame_keypoints[i], frame_idx)
        pred_tracks.append(tracks.squeeze(0))
        pred_vis_scores.append(vis.squeeze(0))
        pred_confs.append(conf.squeeze(0))

        # Get corresponding 3D points
        query_points = query_frame_keypoints[i]
        query_points_round_long = query_points.squeeze(0).long().cpu().numpy()
        query_points_3d = points_3d[frame_idx][
            query_points_round_long[:, 1], query_points_round_long[:, 0]
        ]
        query_points_3d_conf = depth_conf[frame_idx][
            query_points_round_long[:, 1], query_points_round_long[:, 0]
        ]

        pred_points_3d.append(query_points_3d)
        pred_points_conf.append(query_points_3d_conf)

        torch.cuda.empty_cache()


    pred_tracks = torch.cat(pred_tracks, dim=1).cpu().numpy()  # (N, P, 2)
    # Visibility from CoTracker
    pred_vis_scores = torch.cat(pred_vis_scores, dim=1).cpu().numpy()  # (N, P, 1)
    # Confidence from CoTracker
    pred_conf_scores = torch.cat(pred_confs, dim=1).cpu().numpy()  # (N, P, 1)
    pred_world_points = np.concatenate(pred_points_3d, axis=0)  # (P, 3)
    # Depth confidence
    pred_world_points_conf = np.concatenate(pred_points_conf, axis=0)

    return pred_tracks, pred_vis_scores, pred_conf_scores, pred_world_points, pred_world_points_conf


def transform_points_to_global(sequence, transforms, scales, node_id=0, parent_id=None, parent_transform=None):
    predictions = sequence.predictions
    edges = sequence.edges
    curr_node = predictions[node_id]

    # True when the current node is the root node
    is_top_root = (parent_id is None) or (parent_id == node_id)
    # True when current node has no children in the tree
    is_leaf = node_id not in edges

    if parent_transform is None:
        parent_transform = np.eye(4)
        agg_transform = parent_transform
    
    else:
        transform = transforms[parent_id][node_id]
        scale = scales[parent_id][node_id]

        if not isinstance(transform, np.ndarray):
            transform = transform.cpu().numpy()

        sim_transform = np.zeros((4, 4))
        sim_transform[:3, :3] = scale * transform[:3, :3]
        sim_transform[:3, 3] = transform[:3, 3]
        sim_transform[3, 3] = 1

        agg_transform = parent_transform @ sim_transform

    points = curr_node['points']
    ones = np.ones((points.shape[0], 1))
    points = np.concatenate([points, ones], axis=1)  # (P, 4)
    points = points @ agg_transform.T
    points /= points[:, 3:]
    points = points[:, :3]

    curr_node['original_points'] = curr_node['points']
    curr_node['points'] = points
    
    if not is_leaf:
        for child_id in edges[node_id]:
            transform_points_to_global(sequence, transforms, scales, child_id, node_id, agg_transform)
    
    
def merge_track_video(sequence: Sequence, vis_threshold=0.6):
    """
    Merge local tracks into global tracks for the video sequence. Everytime process one of
    the overlapping images and merge the track using nearest neighbor algorithm. 

    sequence: 
      - prediction['tracks']
      - prediction['vis_scores']
      - prediction['points']
    """
    # points_3d: list[(N, H, W, 3)]
    # pred_points_3d: list[(P, 3)]
    overlap = sequence.overlap
    predictions = sequence.predictions
    
    final_track = predictions[0]['tracks']
    final_pred_points_3d = predictions[0]['points']
    final_vis_scores = predictions[0]['vis_scores']

    for i in range(1, len(predictions)):
        # track: (N, P, 2)
        vis_scores = predictions[i]['vis_scores']
        points = predictions[i]['points']
        track = predictions[i]['tracks']

        num_images_before = final_track.shape[0]
        num_images = track.shape[0]

        # Pad final track
        ext_final_track = np.zeros((num_images_before + num_images - overlap, final_track.shape[1], 2))
        ext_final_track[:final_track.shape[0]] = final_track
        final_track = ext_final_track
        # Pad vis score
        ext_vis_scores = np.zeros((num_images_before + num_images - overlap, final_track.shape[1]))
        ext_vis_scores[:final_vis_scores.shape[0]] = final_vis_scores
        final_vis_scores = ext_vis_scores

        for j in range(overlap):
            # Process each overlapped image
            image_track_prev = final_track[num_images_before - overlap + j]  # （P1, 2)
            image_track_curr = track[j]  # (P2, 2)

            # Construct a KDTree for point matching
            tree = KDTree(image_track_curr)
            matches = tree.query_ball_point(image_track_prev, r=0.5)

            # Copy 3d points with the same track in the same image
            valid_mask = np.ones(track.shape[1], dtype=bool)  # Number of points in curr track
            for idx in range(len(matches)):
                # Merge this point to the previous track and delete it
                if len(matches[idx]) > 0 and vis_scores[j][matches[idx][0]] > vis_threshold and final_vis_scores[num_images_before - overlap + j][idx] > vis_threshold:
                    final_track[num_images_before:, idx, :] = track[overlap:, matches[idx][0], :]
                    final_vis_scores[num_images_before:, idx] = vis_scores[overlap:, matches[idx][0]]
                    valid_mask[matches[idx][0]] = False
            
            # Delete redundent 3d points in curr track
            track = track[:, valid_mask, :]
            points = points[valid_mask]
            vis_scores = vis_scores[:, valid_mask]

        # Concat tracks together, assign 0 vis scores to padded points
        ext_final_track = np.zeros((final_track.shape[0], final_track.shape[1] + track.shape[1], 2))
        ext_final_track[:, :final_track.shape[1]] = final_track
        final_track = ext_final_track
        final_track[-track.shape[0]:, -track.shape[1]:] = track

        ext_vis_scores = np.zeros((final_vis_scores.shape[0], final_vis_scores.shape[1] + track.shape[1]))
        ext_vis_scores[:, :final_vis_scores.shape[1]] = final_vis_scores
        final_vis_scores = ext_vis_scores
        final_vis_scores[-track.shape[0]:, -track.shape[1]:] = vis_scores

        final_pred_points_3d = np.concatenate([final_pred_points_3d, points])

    return final_track, final_vis_scores, final_pred_points_3d


def merge_track_graph_tree(sequence: Sequence, query_frames, node_id=0, vis_threshold=0.6, conf_threshold=0.6, radius=1):
    predictions = sequence.predictions
    curr_node = predictions[node_id]
    edges = sequence.edges

    curr_tracks = curr_node["tracks"]
    curr_vis_scores = curr_node["vis_scores"]
    curr_points = curr_node["points"]
    curr_tracker_conf = curr_node["tracker_conf"]
    curr_points_conf = curr_node['points_conf']

    is_leaf = node_id not in edges

    # Extend the local track to match the shape of the final track
    final_track = np.zeros((len(sequence.images), curr_tracks.shape[1], 2))
    final_track[sequence.subset_to_img_ids[node_id]] = curr_tracks
    # Initialize all untracked vis to -1
    final_vis_scores = np.ones((len(sequence.images), curr_tracks.shape[1])) * -1
    final_vis_scores[sequence.subset_to_img_ids[node_id]] = curr_vis_scores

    final_points = curr_points
    final_points_conf = curr_points_conf

    curr_covered_frames = sequence.subset_to_img_ids[node_id]
    curr_covered_frames_set = set(curr_covered_frames)

    # If this is the leaf node then just return its local track in global
    if is_leaf:
        return final_track, final_vis_scores, final_points, final_points_conf, curr_covered_frames

    child_ids = edges[node_id]
    for i, idx in enumerate(child_ids.keys()):
        # Recursion call to process all children
        (
            child_track, 
            child_vis_scores, 
            child_points,
            child_points_conf,
            child_covered_frames
        ) = merge_track_graph_tree(sequence, query_frames, node_id=idx, 
                              vis_threshold=vis_threshold, conf_threshold=conf_threshold)

        
        # parent_track_idx = [sequence.subset_to_img_ids[node_id][query_frames[idx]["from_parent"]]]
        overlap_indices = edges[node_id][idx]["parent_overlap_indices"]
        overlap_indices_global = curr_covered_frames[overlap_indices]
        
        curr_covered_frames_set = curr_covered_frames_set.union(set(child_covered_frames))
        
        # First add track to thoses overlapping images selected for tracking in parent/child
        # TODO: Fix the sequence.subset_to_img_ids issue
        (
            final_track, 
            final_vis_scores, 
            final_points,
            final_points_conf
        ) = add_track(final_track, child_track, 
                        final_vis_scores, child_vis_scores, 
                        final_points, child_points, 
                        final_points_conf, child_points_conf,
                        child_covered_frames,
                        overlap_indices_global, curr_tracker_conf,
                        vis_threshold, conf_threshold,
                        radius=radius)
        

    return final_track, final_vis_scores, final_points, final_points_conf, np.fromiter(curr_covered_frames_set, dtype=np.int64)



def add_track(full_track, new_track, full_vis, new_vis, 
              full_points, new_points, 
              full_points_conf, new_points_conf,
              img_idx_global,
              overlap_idx_global, tracker_conf,
              vis_threshold, conf_threshold, radius=1.0):
    """
    Merge new_track into full_track using nearest neighbor algorithm.

    full_track: (N, P, 2)
    new_track: (N, P1, 2)
    full_vis: (N, P)
    new_vis: (N, P1)
    full_points: (P, 3)
    new_points: (P1, 3)
    tracker_conf: (N, P1)
    img_idx: List[int]

    return: 
    full_track: (N, P, 2)
    """

    total_num_images = full_track.shape[0]
    num_images = len(overlap_idx_global)
    non_overlap_idx_global = [idx for idx in img_idx_global if idx not in overlap_idx_global]

    for j in range(num_images):
        # Process each overlapped image
        track_index = overlap_idx_global[j]
        image_track_prev = full_track[track_index]  # （P1, 2)
        image_track_curr = new_track[track_index]  # (P2, 2)

        # Construct a KDTree for point matching
        tree = KDTree(image_track_curr)
        dists, matches = tree.query(image_track_prev, k=1, distance_upper_bound=radius)
        # Mark unmatched points as -1
        matches[matches == len(image_track_curr)] = -1

        # Copy 3d points with the same track in the same image
        valid_mask = np.ones(new_track.shape[1], dtype=bool)  # Number of points in curr track
        for idx in range(len(matches)):
            # Merge this point to the previous track and delete it
            if matches[idx] != -1 and new_vis[track_index][matches[idx]] > vis_threshold and \
                full_vis[track_index][idx] > vis_threshold:

                full_track[non_overlap_idx_global, idx, :] = new_track[non_overlap_idx_global, matches[idx], :] # TODO: Not overwrite all idx
                full_vis[non_overlap_idx_global, idx] = new_vis[non_overlap_idx_global, matches[idx]]

                # print(full_points[idx], new_points[matches[idx]])
                # full_points[idx] = (full_points_conf[idx] * full_points[idx] + new_points_conf[matches[idx]] * new_points[matches[idx]]) / (full_points_conf[idx] + new_points_conf[matches[idx]])
                valid_mask[matches[idx]] = False
        
        # Delete redundent 3d points in curr track
        new_track = new_track[:, valid_mask, :]
        new_points = new_points[valid_mask]
        new_vis = new_vis[:, valid_mask]
        new_points_conf = new_points_conf[valid_mask]

    # Concat tracks together, assign 0 vis scores to padded points
    ext_full_track = np.zeros((total_num_images, full_track.shape[1] + new_track.shape[1], 2))
    ext_full_track[:, :full_track.shape[1]] = full_track
    final_tracks = ext_full_track
    final_tracks[img_idx_global, -new_track.shape[1]:] = new_track[img_idx_global]

    ext_vis_scores = np.ones((full_vis.shape[0], full_vis.shape[1] + new_vis.shape[1])) * -1
    ext_vis_scores[:, :full_vis.shape[1]] = full_vis
    final_vis_scores = ext_vis_scores
    final_vis_scores[img_idx_global, -new_vis.shape[1]:] = new_vis[img_idx_global]

    final_pred_points_3d = np.concatenate([full_points, new_points])
    final_points_conf = np.concatenate([full_points_conf, new_points_conf])

    return final_tracks, final_vis_scores, final_pred_points_3d, final_points_conf


def merge_track_graph(sequence: Sequence, vis_threshold=0.6, conf_threshold=0.6):
    num_images = sequence.images.shape[0]

    predictions = sequence.predictions
    subset_to_img_ids = deepcopy(sequence.subset_to_img_ids)

    # Convert subset_to_img_ids from tensor to ndarray
    for key in subset_to_img_ids.keys():
        subset_to_img_ids[key] = subset_to_img_ids[key].cpu().numpy()
    
    first_track = predictions[0]['tracks']
    first_points = predictions[0]['points']
    first_vis = predictions[0]['vis_scores']

    full_track = np.zeros((num_images, first_track.shape[1], 2))
    full_points = np.zeros((first_track.shape[1], 3))
    full_vis = np.zeros((num_images, first_track.shape[1]))

    full_track[subset_to_img_ids[0]] = first_track
    full_points[:] = first_points
    full_vis[subset_to_img_ids[0]] = first_vis

    for i in range(1, len(predictions)):
        curr_track = predictions[i]['tracks']
        curr_points = predictions[i]['points']
        curr_vis = predictions[i]['vis_scores']
        curr_tracker_conf = predictions[i]['tracker_conf']
        curr_overlap_ids = subset_to_img_ids[i]

        full_track, full_vis, full_points = add_track(full_track, curr_track, 
                                                      full_vis, curr_vis, 
                                                      full_points, curr_points, 
                                                      curr_overlap_ids, curr_tracker_conf,
                                                      vis_threshold=vis_threshold,
                                                      conf_threshold=conf_threshold)
        
    return full_track, full_vis, full_points


def filter_and_extend_track(model: VGGT, sequence: Sequence, track, vis_scores, points_3d, points_conf,
                             vis_threshold=0.5, valid_frame_frac=0.3, num_queries=20, device="cuda"):
    """
    track: (N, P, 2)
    vis_scores: (N, P)
    """
    track = track.copy()
    vis_scores = vis_scores.copy()
    predictions = sequence.predictions
    num_images = track.shape[0]

    invalid_points = vis_scores < vis_threshold
    # Replace -1 with NaN
    vis_scores_masked = np.where(invalid_points, np.nan, vis_scores)
    avg_vis_scores = np.nanmean(vis_scores_masked, axis=0)  # [P]
    # Filter by average vis scores for valid points
    # vis_mask = avg_vis_scores > vis_threshold
    # track = track[:, vis_mask]
    # vis_scores = vis_scores[:, vis_mask]
    # invalid_points = invalid_points[:, vis_mask]
    # points_3d = points_3d[vis_mask]
    # points_conf = points_conf[vis_mask]

    
    # Filter tracks that are completely invalid in some clusters
    # Or maybe keep them since we make get valid track for those
    # for i, prediction in enumerate(predictions):
    #     img_ids = sequence.subset_to_img_ids[i] 
    #     mask = (vis_scores[img_ids] != -1).sum(0) != 0

    #     track = track[:, mask]
    #     vis_scores = vis_scores[:, mask]
    #     invalid_points = invalid_points[:, mask]

    # TODO: Select the frame using inliers number for max reproj
    scores = invalid_points.sum(0)  # [P]
    sorted_idx = np.argsort(scores)  # ascending

    # The track to be extended
    # if sorted_idx.shape[0] // 2 < num_queries:
    #     query_track_ids = sorted_idx
    # else:
    #     query_track_ids = np.random.choice(sorted_idx[-sorted_idx.shape[0] // 2:], size=num_queries, replace=False)
    query_track_ids = sorted_idx[-num_queries:]

    some_need_track = True
    while some_need_track:
        some_need_track = False
        for i, prediction in enumerate(predictions):
            invalid_points = vis_scores == -1
            img_ids = sequence.subset_to_img_ids[i]
            curr_invalid_points = invalid_points[img_ids][:, query_track_ids]  # [N1, P]
            # Select the frame which has the largest valid point
            query_frame = np.argmin(curr_invalid_points.sum(1))

            query_frame_global = img_ids[query_frame]
            query_points = track[query_frame_global, query_track_ids]  # [num_queries, 2]
            # Filter tracks that are not needed for tracking in this cluster
            need_track = (vis_scores[img_ids][:, query_track_ids] == -1).sum(0) != 0  # [P]
            can_track = vis_scores[query_frame_global, query_track_ids] > 0.3 # [P]
            track_mask = need_track & can_track
            query_points = query_points[track_mask]  # [P]

            if track_mask.sum() == 0:
                continue
            some_need_track = True
            extend_track, vis, conf = predict_track_from_cotracker(model, torch.from_numpy(prediction['features']).to(torch.float).unsqueeze(0).to(device), 
                                                                torch.from_numpy(query_points).to(torch.float).unsqueeze(0).to(device), 
                                                                query_frame)

            track[img_ids[:, None], query_track_ids[track_mask][None, :]] = extend_track.squeeze(0).detach().cpu().numpy()
            vis_scores[img_ids[:, None], query_track_ids[track_mask][None, :]] = vis.squeeze(0).detach().cpu().numpy()

    
    valid_mask = vis_scores > vis_threshold
    filter = valid_mask.sum(0) > num_images * valid_frame_frac

    if not filter.any():
        print("[Warning] Track becomes empty after filtering!")
        return track, vis_scores, None

    track = track[:, filter]
    vis_scores = vis_scores[:, filter]
    points_3d = points_3d[filter]
    points_conf = points_conf[filter]
    
    return track, vis_scores, points_3d, points_conf


def split_tracks_by_image(track: np.ndarray, points: np.ndarray, vis_scores: np.ndarray, thresh: float = 0.5):
    """
    Given:
        track:      (N, P, 3)  — per-image 2D projections (x, y, z or whatever 3 channels)
        points:     (P, 3)     — 3D points corresponding to columns in track
        vis_scores: (N, P, 1)  — visibility/confidence scores per (image, point)
        thresh:     float      — visibility threshold; only keep scores > thresh

    Returns:
        tracks:     list of np.ndarray of shape (P_i, 2) for each image i
        points_id:  list of np.ndarray of shape (P_i,)   for each image i
    """
    N, P, D = track.shape
    if len(vis_scores.shape) == 2:
        vis_scores = np.expand_dims(vis_scores, 2)
    assert D >= 2, f"Expected track last dim >= 2 (x,y,...), got {D}"
    assert vis_scores.shape == (N, P, 1), f"vis_scores must be (N, P, 1), got {vis_scores.shape}"
    assert points.shape[0] == P, "points count must match track’s 2nd dimension"

    tracks = []
    points_id = []

    for i in range(N):
        vis_mask = vis_scores[i, :, 0] > thresh        # (P,)
        xy = track[i, vis_mask, :2]                    # (P_i, 2)
        ids = np.nonzero(vis_mask)[0]                  # (P_i,)
        tracks.append(xy)
        points_id.append(ids)

    return tracks, points_id
