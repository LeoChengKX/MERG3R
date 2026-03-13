from abc import ABC, abstractmethod
import torch
from collections import deque, OrderedDict
import math
import numpy as np

from algos.utils import convert_to_homogeneous_matrix, remove_homogeneous_row, get_sim_matrix
from algos.MST import build_mst
from algos.shortest_path import most_similar_path


class Sequence(ABC):
    """
    Abstract base class for managing an image sequence in 3D reconstruction.

    Handles loading images, partitioning into local subsets, and merging local
    reconstructions into a global coordinate frame. Subclasses must define
    how edges between images are generated and how the sequence is split
    into subsets for local reconstruction.
    """
    def __init__(self, images, image_names, device='cuda'):
        self.images = images
        self.image_names = image_names
        self.device = device

    @abstractmethod
    def generate_edges(self):
        """
        Compute connectivity information between images.

        This method should build a graph indicating which
        subsets (or subsets of images) have sufficient overlap or feature
        matches to be reconstructed together.

        Returns:
            Dictionary of dictionaries with parent_id as the outer key, child_id as the inner key, and edge as the value
            edge = {
                parent_id   : Int,
                child_id    : Int,
                parent_overlap_indices: Indices within the parent node of the images that are also present in the child node,
                child_overlap_indices: Indices within the child node of the images that are also present in the parent node
                }
        """
        pass
    
    @abstractmethod
    def split(self):
        """Partition the full image set into smaller subsets.

        Defines how the sequence is broken into local reconstruction clusters,
        e.g., by temporal order, spatial grouping, or feature-based clustering.

        Returns:
            subset_to_img_ids (Dict[int, List[int]]):
                Mapping from subset index to list of image indices in that subset.
        """
        pass
    
    def change_reference_frame(self, reference_frame_idx = None):
        """Rebase all camera poses to a chosen reference frame.

        Converts the stored extrinsic matrices to be relative to the camera
        pose at `reference_frame_idx`. If not provided, selects the earliest
        image by name.

        Args:
            reference_frame_idx (int, optional):
                Index of the image to serve as the new reference frame.
                Defaults to the earliest-named image in `self.image_names`.
        """
        if reference_frame_idx is None:
            earliest_frame = min([name for name in self.image_names])
            reference_frame_idx = self.image_names.index(earliest_frame)
        
        homogenous_extris = convert_to_homogeneous_matrix(self.merged_predictions['extrinsic'])

        reference_frame_extrinsic_inv = torch.linalg.inv(homogenous_extris[reference_frame_idx])
        new_reference_extris = homogenous_extris @ reference_frame_extrinsic_inv

        self.merged_predictions['extrinsic'] = remove_homogeneous_row(new_reference_extris)
    
    def reorder_images(self):
        """Reorder `self.image_names` to match merged reconstruction order.

        After merging local reconstructions, the order of images in
        `self.predictions['image_ids']` may differ. This method updates
        `self.image_names` accordingly.
        """
        image_names = []
        images = []

        for idx in self.merged_predictions['image_ids']:
            image_names.append(self.image_names[idx])
            images.append(self.images[idx])

        self.image_names = image_names
        self.images = torch.stack(images)

    @staticmethod
    def similarity_transform(extri, transform, scale):
        """Apply a similarity (rotation + translation + scale) transform.

        Args:
            extri (torch.Tensor):
                Tensor of shape (N, 3, 4) containing N extrinsic matrices.
            transform (np.ndarray or torch.Tensor):
                4 x 4 homogeneous transformation matrix.
            scale (float):
                Scale factor to apply to translation components.

        Returns:
            torch.Tensor:
                Transformed extrinsic matrices of shape (N, 3, 4).
        """
        extri = extri.to(device='cuda')
        converted_curr_R = (extri[..., :3] @ transform[:3, :3].T)            # N, 3, 3
        converted_curr_t = scale * extri[:, :3, 3:] - converted_curr_R @ transform[None, :3, 3:] # N, 3, 1
        converted_extri = torch.cat([converted_curr_R, converted_curr_t], dim=-1)

        return converted_extri
    
    def transform_to_shared_frame(self, transforms, scales, node_id=0, parent_id=None):
        """Recursively merge local reconstructions into a shared global frame.

        Traverses the tree of local reconstruction clusters (defined by
        `self.edges`) and applies the given relative transforms and scales
        to produce a single global reconstruction. Updates
        `self.predictions` and `self.image_names`.

        Args:
            transforms (Dict[int, Dict[int, np.ndarray]]):
                Nested mapping from parent cluster ID to child cluster ID to
                4x4 transformation matrices.
            scales (Dict[int, Dict[int, float]]):
                Nested mapping from parent cluster ID to child cluster ID to
                scale factors.
            node_id (int, optional):
                Current cluster ID being processed. Defaults to 0 (root).
            parent_id (int, optional):
                Parent cluster ID. If None, treats `node_id` as the global root.

        Returns:
            Dict[str, np.ndarray or torch.Tensor]:
                Transformed prediction data for the subtree rooted at `node_id`.
        """
        predictions = self.predictions
        edges = self.edges
        curr_node = predictions[node_id]
        transformed_node = {}
        from_numpy_alias = lambda x: torch.from_numpy(x).float().to(device='cpu')

        # True when the current node is the root node
        is_top_root = (parent_id is None) or (parent_id == node_id)
        # True when current node has no children in the tree
        is_leaf = node_id not in edges

        # ---------------- Overlap Handling ----------------
        # Indices of images that are in this cluster as well as parent cluster
        overlap_indices = []
        if not is_top_root: 
            overlap_indices = edges[parent_id][node_id]['child_overlap_indices']

        # Get the non overlapping indices so that we can propogate only the necessary information upwards the tree
        total_len = curr_node['extrinsic'].shape[0]
        total_indices = torch.arange(total_len, device='cpu')

        overlap_indices_tensor = torch.tensor(overlap_indices, dtype=torch.int, device='cpu')
        non_overlap_mask = torch.ones(total_len, dtype=torch.bool, device='cpu')
        non_overlap_mask[overlap_indices_tensor] = False
        non_overlap_indices = total_indices[non_overlap_mask]
        
        # Collect only the information that is in this node but not the parent node
        for key in curr_node.keys():
            if not isinstance(curr_node[key], np.ndarray) and not isinstance(curr_node[key], torch.Tensor): continue
            transformed_node[key] = from_numpy_alias(curr_node[key])[non_overlap_indices]
        # Collect image ids to make image_name reordering easier later
        transformed_node['image_ids'] = (self.subset_to_img_ids[node_id][non_overlap_indices]).to(device='cpu')
        
        # Root is a leaf so there is only one frame and nothing to do
        if is_leaf and is_top_root: return transformed_node
        
        if not is_leaf:
            # ---------------- Recursive Case ----------------
            # Collect each child's subtree extrinsics after transforming them into this node's frame
            for child_id in edges[node_id]:
                converted_child  = self.transform_to_shared_frame(transforms, scales, node_id=child_id, parent_id=node_id)
                for key in converted_child.keys():
                    if key not in ["tracks", "vis_scores", "points", "tracker_conf"]:
                        converted_child[key] = converted_child[key].to(device='cpu')
                        transformed_node[key] = torch.cat([transformed_node[key], converted_child[key]], dim=0)
            # Root is the global frame, no transformations to do from here
            if is_top_root:
                self.merged_predictions = transformed_node
                # Reorder image names to reflect changes after merging
                # self.reorder_images()
                # Make all frames relative to the earliest frame in the sequence
                # self.change_reference_frame()
                return self.merged_predictions

        # Transform entire subtree (including current node) into parent frame
        T = transforms[parent_id][node_id]
        scale = scales[parent_id][node_id]
        subtree_extri = transformed_node['extrinsic']

        converted_extri = self.similarity_transform(subtree_extri, T, scale)

        transformed_node['extrinsic'] = converted_extri
        transformed_node['depth'] *= scale
        intrinsic = transformed_node['intrinsic']
        # intrinsic[:, 0, 0], intrinsic[:, 1, 1] = intrinsic[:, 0, 0] / scale, intrinsic[:, 1, 1] / scale
        transformed_node['intrinsic'] = intrinsic
        
        for key in ["tracks", "vis_scores", "points", "tracker_conf"]:
            transformed_node.pop(key, None)

        return transformed_node
    

class VideoSequence(Sequence):
    """Sequence implementation for temporally ordered video frames.

    Splits a linear list of images (frames) into overlapping, fixed-size
    subsets and constructs edges between consecutive subsets indicating
    their overlapping frame indices. This enables local reconstruction
    per subset followed by hierarchical merging.
    """
    def __init__(self, image_list, image_names, dist_bs, overlap):
        super().__init__(image_list, image_names)
        self.dist_bs = dist_bs
        self.overlap = overlap
        self.split()
        self.generate_edges()

    
    def split(self):
        """Partition the video frames into overlapping subsets.

        Frames are divided into chunks of length `self.dist_bs`. Each
        subsequent chunk starts `self.dist_bs - self.overlap` frames after
        the previous one, creating `self.overlap` shared frames between
        consecutive subsets. Stops when fewer than `overlap` frames would
        remain for the next subset.

        Side Effects:
            Sets the following attributes:
                - image_split (List[Tensor/List]): Frame subsets.
                - names_split (List[List[str]]): Name subsets.
                - num_subsets (int): Number of created subsets.
                - subset_to_img_ids (Dict[int, torch.Tensor]): Mapping from
                  subset index to the global frame indices it contains.
        """
        image_split = []
        names_split = []
        subset_to_img_ids = OrderedDict()
        num_subsets = 0
        current = 0
        while current < len(self.images):   
            image_split.append(self.images[current:current + self.dist_bs])
            names_split.append(self.image_names[current:current + self.dist_bs])
            curr_img_ids = torch.tensor(list(range(min(current, len(self.images)), 
                                                   min(current + self.dist_bs, len(self.images)))
                                                ), dtype=torch.int, device=self.device)
            subset_to_img_ids[num_subsets] = curr_img_ids
            current = current + self.dist_bs - self.overlap
            num_subsets += 1
            if current + self.overlap >= len(self.images):
                break

        # Resplit last two subsets if last subset is less than 50% of subset_size
        if num_subsets >= 2:
            last_subset_size = len(image_split[-1])
            threshold = 0.5 * self.dist_bs
            if last_subset_size < threshold:
                # Get combined indices from last two subsets (they already have overlap)
                second_last_indices = subset_to_img_ids[num_subsets - 2].tolist()
                last_indices = subset_to_img_ids[num_subsets - 1].tolist()
                
                # Combine and remove duplicates while preserving order
                combined_indices = []
                seen = set()
                for idx in second_last_indices + last_indices:
                    if idx not in seen:
                        combined_indices.append(idx)
                        seen.add(idx)
                
                # Calculate equal split size
                total_combined = len(combined_indices)
                # Each subset should have approximately half, accounting for overlap
                split_point = (total_combined + self.overlap) // 2
                
                # Create two equal subsets with overlap
                first_subset_indices = combined_indices[:split_point]
                second_subset_indices = combined_indices[split_point - self.overlap:]
                
                # Replace last two subsets
                image_split[-2] = torch.stack([self.images[i] for i in first_subset_indices])
                image_split[-1] = torch.stack([self.images[i] for i in second_subset_indices])
                names_split[-2] = [self.image_names[i] for i in first_subset_indices]
                names_split[-1] = [self.image_names[i] for i in second_subset_indices]
                subset_to_img_ids[num_subsets - 2] = torch.tensor(first_subset_indices, dtype=torch.int, device=self.device)
                subset_to_img_ids[num_subsets - 1] = torch.tensor(second_subset_indices, dtype=torch.int, device=self.device)

        self.image_split = image_split
        self.names_split = names_split
        self.num_subsets = num_subsets
        self.subset_to_img_ids = subset_to_img_ids

    def generate_edges(self):
        """Create directed edges between consecutive overlapping subsets.

        For each adjacent subset pair `(i, i+1)`, constructs an edge
        describing which frame indices overlap. Overlap indices are
        expressed relative to each subset
        """
        edges = {}
        for i in range(self.num_subsets - 1):
            edge = {
                "parent": i,
                "child": i + 1,
                "parent_overlap_indices": [i for i in range(-self.overlap, 0)],
                "child_overlap_indices": list(range(self.overlap)),

            }
            if i not in edges: edges[i] = {}
            edges[i][i+1] = edge
        
        self.edges = edges


class GraphSequence(Sequence):
    """Sequence implementation based on feature-space graph clustering.

    Uses a vision backbone (e.g., DINOv2) to extract feature descriptors for
    all images, builds a k-NN feature graph, and clusters that graph into
    overlapping subsets. The resulting clusters form a tree (via an MST/BFS
    traversal) whose edges encode image overlap for later merging.
    """
    def __init__(self, image_list, image_names, max_cluster_size=55, overlap=10):
        super().__init__(image_list, image_names)
        self.max_cluster_size = max_cluster_size
        sim_matrix, feats = get_sim_matrix(self.images, return_feats=True)
        clusters, adjacency, overlaps, result = build_mst(sim_matrix, feats, max_children=3, max_cluster_size=max_cluster_size, num_overlaps=overlap, min_sim=0.0)
        self.clusters=result
        self.generate_edges()
        self.split()
        
        del feats, sim_matrix

    
    def split(self):
        """Group clustered image indices into subsets for local reconstruction.

        For each cluster produced by the graph clustering step, collects its
        list of image IDs—including enforced overlaps and builds tensors of images and names.
        """
        image_split = []
        names_split = []
        subset_to_img_ids = OrderedDict()

        for cluster in self.clusters.values():
            # mst function already added overlapping images into the set
            indices = cluster.image_ids

            subset_images = torch.stack([self.images[i] for i in indices])
            subset_names = [self.image_names[i] for i in indices]

            image_split.append(subset_images)
            names_split.append(subset_names)
            
            subset_to_img_ids[cluster.cluster_id] = torch.tensor(indices, dtype=torch.int, device=self.device)


        self.image_split = image_split
        self.names_split = names_split
        self.num_subsets = len(self.clusters)
        self.subset_to_img_ids = subset_to_img_ids


    def generate_edges(self):
        """Construct a tree of cluster overlaps via BFS/MST expansion.

        Performs a breadth-first traversal starting from the first cluster to
        create directed edges (parent → child). For each edge, overlapping
        image IDs are unified into both clusters' `image_ids` lists, and the
        indices of those overlaps relative to each cluster are recorded.
        """
        # MST Algorithm
        q = deque([self.clusters[0]])
        self.root_idx = float('inf')
        edges = {}
        visited = set()

        while q:
            curr = q.popleft()
            parent_id = curr.cluster_id
            visited.add(parent_id)
            for child_id in curr.overlaps:
                if child_id in visited: continue
                neighbor = self.clusters[child_id]
                q.append(neighbor)
                
                # Add non-duplicate overlapping images
                overlapping_images = set(curr.overlaps[child_id]) | set(neighbor.overlaps[parent_id])
                new_curr_images = list(overlapping_images - set(curr.image_ids))
                new_neighbor_images = list(overlapping_images - set(neighbor.image_ids))
                curr.image_ids += new_curr_images
                neighbor.image_ids += new_neighbor_images

                parent_overlap_indices = [
                    curr.image_ids.index(img_id)
                    for img_id in overlapping_images
                ]
                child_overlap_indices = [
                    neighbor.image_ids.index(img_id)
                    for img_id in overlapping_images
                ]

                assert len(child_overlap_indices) > 0 and len(parent_overlap_indices) == len(child_overlap_indices)
                # Edge: parent -> Child 
                edge = {
                    "parent": parent_id,
                    "child": child_id,
                    "parent_overlap_indices": parent_overlap_indices,
                    "child_overlap_indices": child_overlap_indices,

                }
                if parent_id not in edges: edges[parent_id] = {}
                edges[parent_id][child_id] = edge
                visited.add(child_id)
        print("[CLUSTERING] Cluster sizes: ", [len(self.clusters[i].image_ids) for i in range(len(self.clusters))])
        assert(len(visited) == len(self.clusters))
        assert(max([len(self.clusters[i].image_ids) for i in range(len(self.clusters))]) <= self.max_cluster_size)
        assert len(set(x for sublist in [c.image_ids for c in self.clusters.values()] for x in sublist)) == len(self.images)
        self.edges = edges


class ShortestPath(Sequence):
    def __init__(self, image_list, image_names, subset_size=55, overlap=10, interleave=2, save_path="", alpha=0.3, splitting_type="interleave"):
        super().__init__(image_list, image_names)

        self.max_images = subset_size
        self.overlap = overlap
        self.interleave = interleave
        self.save_path = save_path
        self.splitting_type = splitting_type
        self.sim_matrix = get_sim_matrix(self.images, alpha=alpha)
        self.path = most_similar_path(self.sim_matrix)
        self.split()
        self.generate_edges()
        
        del self.sim_matrix
        print("SUBSET SIZES: ", [len(i) for i in self.image_split])



    def split(self):
        """Partition the video frames into overlapping subsets.

        Frames are divided into chunks of length `self.dist_bs`. Each
        subsequent chunk starts `self.dist_bs - self.overlap` frames after
        the previous one, creating `self.overlap` shared frames between
        consecutive subsets. Stops when fewer than `overlap` frames would
        remain for the next subset.

        Side Effects:
            Sets the following attributes:
                - image_split (List[Tensor/List]): Frame subsets.
                - names_split (List[List[str]]): Name subsets.
                - num_subsets (int): Number of created subsets.
                - subset_to_img_ids (Dict[int, torch.Tensor]): Mapping from
                  subset index to the global frame indices it contains.
        """
        self.video_path = self.path
        # # Number of whole subsets
        num_subsets = math.floor(len(self.path) / self.max_images)
        new_path = []

        if self.splitting_type == "interleave": 

            for i in range(num_subsets):
                idx = i
                while idx < len(self.path):
                    new_path.append(self.path[idx])
                    idx += num_subsets
        
        elif self.splitting_type == "zigzag":
        
            forward = True
            idx = 0
            for _ in range(self.interleave):
                if forward:
                    while idx < len(self.path):
                        new_path.append(self.path[idx])
                        idx += self.interleave

                    idx -= self.interleave + 1
                    forward = False
                
                else:
                    while idx > 0:
                        new_path.append(self.path[idx])
                        idx -= self.interleave
                    
                    idx += self.interleave - 1
                    forward = True

            tail = self.path[len(new_path):]

            # Inserting tail images before
            for img in tail:
                max_sim = 0.0
                max_idx = 0
                for i in range(len(new_path) - 1):
                    sim_score = (self.sim_matrix[new_path[i]][img] + self.sim_matrix[new_path[i + 1]][img]) / 2
                    if sim_score > max_sim:
                        max_sim = sim_score
                        max_idx = i
                
                new_path.insert(max_idx + 1, img)
        
        
        elif self.splitting_type == "threshold":
            picked = torch.zeros(len(self.path))
            while torch.sum(picked) != len(self.path):
                count = 0
                start = torch.where(picked == 0)[0][0].item()
                
                new_path.append(self.path[start])
                picked[start] = 1

                old_start = start
                
                for i in range(old_start, len(self.path)):
                    threshold1 = torch.quantile(self.sim_matrix[start], 1.00)
                    threshold2 = torch.quantile(self.sim_matrix[start], 0.50)

                    if (not picked[i]) and (self.sim_matrix[self.path[start]][self.path[i]] <= threshold1) and (self.sim_matrix[self.path[start]][self.path[i]] >= threshold2):
                        new_path.append(self.path[i])
                        picked[i] = 1
                        start = i
                    
                        count += 1

        elif self.splitting_type == "original":
            new_path = list(range(0, len(self.path)))

        elif self.splitting_type == "original_threshold":
            self.path = list(range(0, len(self.path)))

            picked = torch.zeros(len(self.path))
            while torch.sum(picked) != len(self.path):
                count = 0
                start = torch.where(picked == 0)[0][0].item()
                
                new_path.append(self.path[start])
                picked[start] = 1

                old_start = start
                
                for i in range(old_start, len(self.path)):
                    threshold1 = torch.quantile(self.sim_matrix[start], 1.00)
                    threshold2 = torch.quantile(self.sim_matrix[start], 0.50)

                    if (not picked[i]) and (self.sim_matrix[self.path[start]][self.path[i]] <= threshold1) and (self.sim_matrix[self.path[start]][self.path[i]] >= threshold2):
                        new_path.append(self.path[i])
                        picked[i] = 1
                        start = i
                    
                        count += 1
            
        else:
            raise ValueError(f"Unknown splitting type: {self.splitting_type}")


        self.path = new_path

        image_split = []
        names_split = []
        subset_to_img_ids = {}
        num_subsets = 0
        current = 0
        while current < len(self.path):
            indices = self.path[current:current + self.max_images]

            subset_images = torch.stack([self.images[i] for i in indices])
            subset_names = [self.image_names[i] for i in indices]

            curr_img_ids = torch.tensor(indices, dtype=torch.int, device=self.device)
            image_split.append(subset_images)
            names_split.append(subset_names)

            subset_to_img_ids[num_subsets] = curr_img_ids
            current = current + self.max_images - self.overlap
            num_subsets += 1
            if current + self.overlap >= len(self.path):
                break

        # Resplit last two subsets if last subset is less than 50% of subset_size
        if num_subsets >= 2:
            last_subset_size = len(image_split[-1])
            threshold = 0.5 * self.max_images
            if last_subset_size < threshold:
                # Get combined path indices from last two subsets
                second_last_indices = subset_to_img_ids[num_subsets - 2].tolist()
                last_indices = subset_to_img_ids[num_subsets - 1].tolist()
                
                # Combine and remove duplicates while preserving order
                combined_indices = []
                seen = set()
                for idx in second_last_indices + last_indices:
                    if idx not in seen:
                        combined_indices.append(idx)
                        seen.add(idx)
                
                # Calculate equal split size
                total_combined = len(combined_indices)
                # Each subset should have approximately half, accounting for overlap
                split_point = (total_combined + self.overlap) // 2
                
                # Create two equal subsets with overlap
                first_subset_indices = combined_indices[:split_point]
                second_subset_indices = combined_indices[split_point - self.overlap:]
                
                # Replace last two subsets
                image_split[-2] = torch.stack([self.images[i] for i in first_subset_indices])
                image_split[-1] = torch.stack([self.images[i] for i in second_subset_indices])
                names_split[-2] = [self.image_names[i] for i in first_subset_indices]
                names_split[-1] = [self.image_names[i] for i in second_subset_indices]
                subset_to_img_ids[num_subsets - 2] = torch.tensor(first_subset_indices, dtype=torch.int, device=self.device)
                subset_to_img_ids[num_subsets - 1] = torch.tensor(second_subset_indices, dtype=torch.int, device=self.device)

        self.image_split = image_split
        self.names_split = names_split
        self.num_subsets = num_subsets
        self.subset_to_img_ids = subset_to_img_ids

    def generate_edges(self):
        """Create directed edges between consecutive overlapping subsets.

        For each adjacent subset pair `(i, i+1)`, constructs an edge
        describing which frame indices overlap. Overlap indices are
        expressed relative to each subset
        """
        edges = {}
        for i in range(self.num_subsets - 1):
            edge = {
                "parent": i,
                "child": i + 1,
                "parent_overlap_indices": [i for i in range(-self.overlap, 0)],
                "child_overlap_indices": list(range(self.overlap)),

            }
            if i not in edges: edges[i] = {}
            edges[i][i+1] = edge
        
        self.edges = edges

def create_sequence(image_list, image_names, subset_size=None, overlap=None, sequence_type=None, save_path="", alpha=0.3, splitting_type="interleave"):
    if sequence_type == 'video':
        assert image_list is not None and subset_size is not None and overlap is not None and image_names is not None
        return VideoSequence(image_list, image_names, subset_size, overlap)

    elif sequence_type == 'graph':
        return GraphSequence(image_list, image_names, overlap=overlap)
    elif sequence_type == "shortest_path":
        return ShortestPath(image_list, image_names, subset_size=subset_size, overlap=overlap, save_path=save_path, alpha=alpha, splitting_type=splitting_type)

    else:
        raise ValueError(f"Uknown sequence type {sequence_type}")