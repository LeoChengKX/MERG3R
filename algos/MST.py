import torch
from dataclasses import dataclass, field
from typing import Dict, List

# ------------------ ImageCluster ------------------

@dataclass
class ImageCluster:
    cluster_id: int
    image_ids: List[int] = field(default_factory=list)            # core images only
    overlaps: Dict[int, List[int]] = field(default_factory=dict)  # neighbor_id -> list of overlap image ids

# ------------------ Union-Find (DSU) ------------------

class UnionFind:
    def __init__(self, n):
        self.p = list(range(n))
        self.r = [0] * n

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.r[ra] < self.r[rb]:
            self.p[ra] = rb
        elif self.r[ra] > self.r[rb]:
            self.p[rb] = ra
        else:
            self.p[rb] = ra
            self.r[ra] += 1
        return True
    
# ------------------ Debugging helper ------------------
def coverage_score(nums):
    a=sorted(set(nums))
    if len(a)<2: return 0.0
    R=a[-1]-a[0]
    if R==0: return 1.0
    g=[max(0,b-a-1) for a,b in zip(a,a[1:])]
    return 1-0.5*((sum(g)/R)+(sum(x>0 for x in g)/(len(a)-1)))

# ------------------ Algo steps ------------------

def split(N, sims, feats, max_cluster_size, max_children, overlaps):
    upper_bound = max_cluster_size - (overlaps * (max_children + 1))
    assert upper_bound > 0, "Clustering configurations are not possible. Try lowering max_children or overlaps, or increasing max_cluster_size"
    clusters = []
    clusters.append([i for i in range(N)])

    def cluster_sizes():
        return [len(c) for c in clusters]
    
    def find_diverse_seeds(cluster):
        cluster_feats = feats[cluster]
        cluster_sims = cluster_feats @ cluster_feats.t()

        # Step 1: Find s1. We can set diagonal to 0 or 1 here,
        # it won't affect the average much, but 0 is standard.
        cluster_sims.fill_diagonal_(0)
        avg_sims = torch.mean(cluster_sims, dim=1)
        s1_local_idx = torch.argmin(avg_sims).item()

        # Step 2: Find s2.
        # As you correctly pointed out, set the diagonal to 1 (max similarity)
        # to ensure argmin never picks s1 as its own most dissimilar point.
        cluster_sims.fill_diagonal_(1)
        s2_local_idx = torch.argmin(cluster_sims[s1_local_idx]).item()

        # The function returns LOCAL indices. The calling `split` function
        # is responsible for mapping them to global indices.
        return (s1_local_idx, s2_local_idx)

    
    while max(cluster_sizes()) > upper_bound:
        # TODO: Add tie breaker
        clusters.sort(key= lambda x: len(x), reverse=True)
        curr = clusters[0]

        # for c in clusters:
        #     c.sort()
            
        clusters.pop(0)

        s1_local_idx, s2_local_idx = find_diverse_seeds(curr)
        s1_idx, s2_idx = curr[s1_local_idx], curr[s2_local_idx] # Map to global


        # hyperparams
        DELTA = 0.1   # if |sim1 - sim2| >= DELTA, assign by similarity only
        BETA  = 0.5    # strength of size-balancing when it's a toss-up

        new_cluster1, new_cluster2 = [s1_idx], [s2_idx]
        N = max(1, len(curr))  # for normalization

        for idx in curr:
            if idx == s1_idx or idx == s2_idx: 
                continue
            sim1, sim2 = sims[idx, s1_idx], sims[idx, s2_idx]
            margin = sim1 - sim2

            if abs(margin) >= DELTA:
                (new_cluster1 if margin > 0 else new_cluster2).append(idx)
            else:
                # bias toward the smaller cluster; positive when c1 is smaller
                bias = BETA * (len(new_cluster2) - len(new_cluster1)) / N
                if (sim1 + bias) > (sim2 - bias): new_cluster1.append(idx)
                else:                              new_cluster2.append(idx)
        
        clusters.append(new_cluster1)
        clusters.append(new_cluster2)

    # --- Merge Clusters ---
    
    return clusters


def split_w_min(N, sims, feats, max_cluster_size, max_children, overlaps, min_sim=0.9, sim_match_ratio=0.5):
    upper_bound = max_cluster_size - (overlaps * (max_children + 1))
    assert upper_bound > 0, "Clustering configurations are not possible. Try lowering max_children or overlaps, or increasing max_cluster_size"
    clusters = []
    clusters.append([i for i in range(N)])

    def cluster_sizes():
        return [len(c) for c in clusters]
    
    def find_diverse_seeds(cluster):
        cluster_feats = feats[cluster]
        cluster_sims = cluster_feats @ cluster_feats.t()

        # Step 1: Find s1. We can set diagonal to 0 or 1 here,
        # it won't affect the average much, but 0 is standard.
        cluster_sims.fill_diagonal_(0)
        avg_sims = torch.mean(cluster_sims, dim=1)
        s1_local_idx = torch.argmin(avg_sims).item()

        # Step 2: Find s2.
        # As you correctly pointed out, set the diagonal to 1 (max similarity)
        # to ensure argmin never picks s1 as its own most dissimilar point.
        cluster_sims.fill_diagonal_(1)
        s2_local_idx = torch.argmin(cluster_sims[s1_local_idx]).item()

        # The function returns LOCAL indices. The calling `split` function
        # is responsible for mapping them to global indices.
        return (s1_local_idx, s2_local_idx)

    global_outliers = []
    while max(cluster_sizes()) > upper_bound:
        # TODO: Add tie breaker
        clusters.sort(key= lambda x: len(x), reverse=True)
        curr = clusters[0]

        # for c in clusters:
        #     c.sort()

        clusters.pop(0)

        s1_local_idx, s2_local_idx = find_diverse_seeds(curr)
        s1_idx, s2_idx = curr[s1_local_idx], curr[s2_local_idx] # Map to global


        # hyperparams
        DELTA = 0.1   # if |sim1 - sim2| >= DELTA, assign by similarity only
        BETA  = 0.5    # strength of size-balancing when it's a toss-up

        new_cluster1, new_cluster2 = [s1_idx], [s2_idx]
        curr_N = max(1, len(curr))  # for normalization
        local_outliers = []
        for idx in curr + global_outliers:
            if idx == s1_idx or idx == s2_idx: 
                continue
            sim1, sim2 = sims[idx, s1_idx], sims[idx, s2_idx]
    
            if torch.max(sim1, sim2) < min_sim:
                if idx not in global_outliers: local_outliers.append(idx)
                continue

            if idx in global_outliers:
                global_outliers.remove(idx)
            assert torch.max(sim1, sim2) >= min_sim
            margin = sim1 - sim2

            if sim1 > sim2:
                assert sim1 >= min_sim
                new_cluster1.append(idx)
            else:
                assert sim2 >= min_sim
                new_cluster2.append(idx)

        clusters.append(new_cluster1)
        clusters.append(new_cluster2)
        global_outliers.extend(local_outliers)
    
    # --- Distribute outliers ---
    for o in global_outliers:
        o_feat = feats[o]
        best_c, best_match_ratio = None, 0
        fallback_c, best_mean_sim = None, 0
        for i in range(len(clusters)):
            cluster_feats = feats[clusters[i]]
            cross_sims = cluster_feats @ o_feat.t()
            match_ratio = torch.sum(cross_sims > min_sim) / cross_sims.shape[0]
            mean_sim = torch.mean(cross_sims)
            if match_ratio > best_match_ratio:
                best_c = i
                best_match_ratio = match_ratio
            if mean_sim > best_mean_sim:
                fallback_c = i
                best_mean_sim = mean_sim

        if best_match_ratio >= sim_match_ratio:
            clusters[best_c].append(o)
        else:
            clusters[fallback_c].append(o)
        # global_outliers.remove(o)

    # --- Merge where possible ---
    def merge_score(c1, c2):
        cross_sims = feats[c1] @ feats[c2].t()
        match_ratio = (cross_sims >= min_sim).float().mean().item()
        return match_ratio if match_ratio >= sim_match_ratio else 0
    
    clusters.sort(key=len)
    merged = True
    while merged:
        merged = False
        for i in range(len(clusters)):
            if merged: break
            best_j, best_score = None, 0
            for j in range(i + 1, len(clusters)):
                if len(clusters[i]) + len(clusters[j]) > upper_bound:
                    continue
                score = merge_score(clusters[i], clusters[j])
                if score > best_score:
                    best_j, best_score = j, score
            
            if best_j is not None:
                clusters[i].extend(clusters[best_j])
                clusters.pop(best_j)
                merged = True


    return clusters

def create_mst(clusters, feats, max_children, use_max_spanning=True, min_sim=0.9):
    """
    Builds a degree-constrained MST over clusters using similarities as weights.
    Returns:
      selected_edges: list of ((i, j), weight) chosen for the tree (len = n - 1)
      adjacency: dict[int, list[tuple[int, float]]] undirected adjacency with weights
      degrees: list[int] degree of each node in the resulting tree
    """
    n = len(clusters)
    if n == 0:
        return [], {}, []
    if n == 1:
        return [], {0: []}, [0]
    # Quick feasibility check: a tree on n nodes needs sum(deg)=2(n-1) and max degree ≤ max_children
    # If max_children == 1, only n <= 2 is feasible.
    if max_children < 1 or (max_children == 1 and n > 2):
        raise RuntimeError(
            f"Degree constraint max_children={max_children} makes a tree on n={n} nodes impossible."
        )

    # Compute cluster-average features and all pairwise edges with weights (similarities)
    cluster_feats = [feats[c] for c in clusters]
    cluster_avg_feat = [torch.mean(feats[c], dim=0) for c in clusters]
    all_edges = []

    for i in range(n):
        for j in range(i + 1, n):
            # Compute cross-similarity matrix
            cross_sims = cluster_feats[i] @ cluster_feats[j].t()
        
            # Use robust statistic instead of min/max
            # percentile_90 = torch.quantile(cross_sims.flatten(), min_sim)
            # w = percentile_90.item()
            
            good_edges = (cross_sims >= min_sim).flatten().to(torch.int)
            w = torch.mean(good_edges.float())

            

            all_edges.append(((i, j), w))

    # Sort edges: max-spanning (desc) for similarities by default; set use_max_spanning=False for min
    all_edges.sort(key=lambda e: e[1], reverse=use_max_spanning)

    uf = UnionFind(n)
    degrees = [0] * n
    selected_edges = []

    for (i, j), w in all_edges:
        # Enforce degree cap before attempting to connect
        if degrees[i] >= max_children or degrees[j] >= max_children:
            continue
        # Skip if already in same component
        if uf.find(i) == uf.find(j):
            continue
        # Add edge
        if uf.union(i, j):
            selected_edges.append(((i, j), w))
            degrees[i] += 1
            degrees[j] += 1
            if len(selected_edges) == n - 1:
                break

    if len(selected_edges) != n - 1:
        # Couldn’t connect all components without violating degree cap
        raise RuntimeError(
            "Could not build a degree-constrained MST under max_children="
            f"{max_children}. Try increasing max_children or relaxing the constraint."
        )

    # Build adjacency for convenience
    adjacency = {i: [] for i in range(n)}
    for (i, j), w in selected_edges:
        adjacency[i].append(j)
        adjacency[j].append(i)

    return selected_edges, adjacency, degrees



def create_overlaps(adjacency, clusters, sims, num_overlaps, min_sim=0.9):
    # get num_overlaps images in donor cluster that have the highest average similarity with reciever cluster
    def get_best_bridge_images(donor_cluster, reciever_cluster):
        candidates = []
        for idx_d in donor_cluster:
            cross_cluster_sim = [sims[idx_d, idx_r] for idx_r in reciever_cluster]
            cross_cluster_avg_sim = sum(cross_cluster_sim) / len(cross_cluster_sim)
            candidates.append((idx_d, cross_cluster_avg_sim))
        
        filtered_candidates = [c for c in candidates if c[1] >= min_sim]
        if len(filtered_candidates) > 0:
            candidates = filtered_candidates
        candidates.sort(key = lambda x: x[1], reverse=True)
        return [c[0] for c in candidates[:num_overlaps]]
    
    #overlap_dict[reciever][donor]
    overlap_dict = {i: {} for i in range(len(clusters))}
    processed_edges = set()
    for parent in adjacency:
        parent_cluster = clusters[parent]
        for child in adjacency[parent]:            
            edge = tuple(sorted([parent, child]))
            if edge in processed_edges:
                continue
            processed_edges.add(edge)
            child_cluster = clusters[child]

            
            parent_to_child_overlaps = get_best_bridge_images(parent_cluster, child_cluster)
            child_to_parent_overlaps = get_best_bridge_images(child_cluster, parent_cluster)

            overlap_dict[parent][child] = child_to_parent_overlaps
            overlap_dict[child][parent] = parent_to_child_overlaps
        
    return overlap_dict
    

def assemble_result(clusters, overlaps):
    result: Dict[int, ImageCluster] = {c: ImageCluster(cluster_id=c) for c in range(len(clusters))}
    for c in range(len(clusters)):
        result[c].image_ids = clusters[c]
        for child in overlaps[c].keys():
            result[c].overlaps[child] = overlaps[c][child]
    
    return result

def pad_clusters(image_clusters: Dict[int, "ImageCluster"], feats: torch.Tensor, max_cluster_size: int, k_outliers: int = 5, min_sim=0.9):
    """
    For each cluster:
      1) Identify up to k_outliers core images with the *lowest* average similarity to the cluster (outliers).
      2) From neighbor clusters, consider candidate images not already in this cluster.
      3) Score each candidate by the *average* similarity to the outliers (not max).
      4) Add the best candidates (descending avg score) up to remaining space into this cluster's overlaps.

    Assumptions:
      - feats: (N, D) torch.Tensor; rows correspond to image IDs; dot product is a similarity.
      - image_clusters[c].image_ids: iterable of ints (core image IDs).
      - image_clusters[c].overlaps: dict[int, list[int]] mapping neighbor cluster id -> overlapped img IDs.
    """

    def find_k_outliers(cluster_img_ids: List[int], k: int) -> List[int]:
        """Return up to k image IDs with *lowest* average similarity to the rest of the cluster."""
        if not cluster_img_ids:
            return []

        idx_tensor = torch.tensor(cluster_img_ids, dtype=torch.long, device=feats.device)
        cluster_feats = feats.index_select(0, idx_tensor)           # (C, D)
        sims = cluster_feats @ cluster_feats.t()                    # (C, C)
        sims.fill_diagonal_(0)                                      # remove self-influence
        avg_sim = sims.mean(dim=0)                                  # (C,)
        # lower avg_sim => more outlier-like
        k = min(k, idx_tensor.shape[0])
        order = torch.argsort(avg_sim, descending=False)[:k]        # pick lowest
        return [cluster_img_ids[i] for i in order.tolist()]

    for c in image_clusters:
        cluster = image_clusters[c]

        # Gather images already present (core + overlaps) to avoid duplicates
        images = set(cluster.image_ids)
        for _, ov_ids in cluster.overlaps.items():
            images.update(ov_ids)

        remaining_space = max_cluster_size - len(images)
        if remaining_space <= 0:
            continue

        core_ids = list(cluster.image_ids)
        if not core_ids:
            continue

        # 1) Outliers in this cluster
        outliers = find_k_outliers(core_ids, k_outliers)
        if not outliers:
            continue

        outlier_idx = torch.tensor(outliers, dtype=torch.long, device=feats.device)
        outlier_feats = feats.index_select(0, outlier_idx)          # (Ko, D)

        # 2) Candidates from neighbor clusters (their cores), excluding already-present imgs
        candidates = []  # (neighbor_id, img_id, avg_score)
        seen = set()

        for neighbor_id in list(cluster.overlaps.keys()):
            neighbor_cluster = image_clusters.get(neighbor_id)
            if neighbor_cluster is None:
                continue

            for img_id in neighbor_cluster.image_ids:
                if img_id in images or img_id in seen:
                    continue
                seen.add(img_id)

                # 3) Average similarity to the outliers
                cand_feat = feats[img_id]                           # (D,)
                sims_to_outliers = outlier_feats @ cand_feat        # (Ko,)
                avg_score = float(sims_to_outliers.mean().item())
                candidates.append((neighbor_id, img_id, avg_score))

        if not candidates:
            continue

        # Sort by average similarity to outliers (desc)
        candidates = [c for c in candidates if c[2] >= min_sim]
        candidates.sort(key=lambda x: x[2], reverse=True)

        # 4) Add best candidates up to remaining_space
        added = 0
        for neighbor_id, img_id, _ in candidates:
            if added >= remaining_space or len(images) >= max_cluster_size:
                break
            if img_id in images:
                continue

            if neighbor_id not in cluster.overlaps:
                cluster.overlaps[neighbor_id] = []
            if img_id not in cluster.overlaps[neighbor_id]:
                cluster.overlaps[neighbor_id].append(img_id)
                images.add(img_id)
                added += 1



def pad_clusters_old(image_clusters: Dict[int, ImageCluster], feats, max_cluster_size):
    def avg_cluster_sim(cluster_img_ids, target_img_id):
        cluster_feats = feats[cluster_img_ids]
        target_feat = feats[target_img_id]
        
        # Compute similarities between target and all cluster images
        similarities = cluster_feats @ target_feat
        avg_sim = torch.mean(similarities)
        return avg_sim.item()

    for c in image_clusters:
        cluster = image_clusters[c]
        
        # Get all images currently in this cluster (core + overlaps)
        images = set(cluster.image_ids)
        for neighbor in cluster.overlaps:
            images = images.union(set(cluster.overlaps[neighbor]))
        
        remaining_space = max_cluster_size - len(images)
        
        if remaining_space > 0:
            # Collect candidate images from neighbor clusters
            candidates = []
            for neighbor in cluster.overlaps:
                neighbor_cluster = image_clusters[neighbor]
                for img in neighbor_cluster.image_ids:
                    if img not in images:  # Don't add images already in current cluster
                        sim_score = avg_cluster_sim(cluster.image_ids, img)
                        candidates.append((neighbor, img, sim_score))
            
            # Sort by similarity (highest first)
            candidates.sort(key=lambda x: x[2], reverse=True)
            
            # Add best candidates up to remaining space
            added = 0
            for neighbor, img, sim_score in candidates:
                if added >= remaining_space:
                    break
                
                # Add to overlaps for the source neighbor
                if neighbor not in cluster.overlaps:
                    cluster.overlaps[neighbor] = []
                cluster.overlaps[neighbor].append(img)
                added += 1




# ------------------ Main ------------------
def build_mst(sims, feats, max_cluster_size=55, num_overlaps=10, max_children=3, min_sim=0.9):

    N = feats.shape[0]
    
    # clusters = split(N, sims, feats, max_cluster_size=max_cluster_size, overlaps=num_overlaps, max_children=max_children)
    clusters = split_w_min(N, sims, feats, max_cluster_size=max_cluster_size, overlaps=num_overlaps, max_children=max_children, min_sim=min_sim)
    selected_edges, adjacency, degrees = create_mst(clusters, feats, max_children, min_sim=min_sim)
    overlaps = create_overlaps(adjacency, clusters, sims, num_overlaps=num_overlaps, min_sim=min_sim)
    result = assemble_result(clusters, overlaps)
    pad_clusters(result, feats, max_cluster_size, min_sim=min_sim)
    # pad_clusters_old(result, feats, max_cluster_size)

    return clusters, adjacency, overlaps, result
    # return result    