import torch
from typing import List, Tuple, Optional

# from __future__ import annotations
import math, random
from typing import List, Dict, Tuple, Optional

@torch.no_grad()
def interleaved_multi_path_torch(
    M: torch.Tensor,
    n_paths: int = 16,
    start_indices: Optional[List[int]] = None,
    polish_top_k: int = 1,
    dedup_starts: bool = True,
) -> Tuple[List[int], float, List[Tuple[List[int], float]]]:
    """
    Find multiple Hamiltonian paths simultaneously by interleaving construction.
    At each step, extend all N paths by one node instead of completing one path at a time.

    Args:
        M: (N,N) similarity matrix
        n_paths: Number of paths to construct simultaneously
        start_indices: Optional explicit starting nodes. If None, uses random unique starts.
        polish_top_k: How many of the top-scoring paths (by greedy weight) to polish with 2-opt.
                      Default 1 (maintains previous behavior of polishing only the best).
                      Use 0 to skip polishing altogether.
        dedup_starts: If True, deduplicate provided start_indices while preserving order
                      and backfill with random unique nodes if needed.

    Returns:
        best_path: Best (possibly polished) path found
        best_weight: Weight of the best (possibly polished) path
        all_results: List of all (path, greedy_weight) results prior to polishing
    """
    N = M.size(0)
    n_paths = max(1, min(n_paths, N))  # Ensure 1 <= n_paths <= N

    # Determine starting points
    if start_indices is not None:
        if dedup_starts:
            # Dedup while preserving order
            seen = set()
            starts = []
            for s in start_indices:
                if 0 <= s < N and s not in seen:
                    starts.append(int(s))
                    seen.add(int(s))
                if len(starts) >= n_paths:
                    break
            # Backfill with random unique starts if needed
            if len(starts) < n_paths:
                remaining = [i for i in range(N) if i not in seen]
                if remaining:
                    rand_perm = torch.tensor(remaining)[torch.randperm(len(remaining))].tolist()
                    starts.extend(rand_perm[: n_paths - len(starts)])
                # If still short (e.g., empty remaining), just truncate to what we have
        else:
            starts = [int(s) for s in start_indices[:n_paths]]
            # If caller passed duplicates and dedup is off, that's intentional.
    else:
        # Random unique starts
        starts = torch.randperm(N)[:n_paths].tolist()

    # Initialize all paths
    paths = [[start] for start in starts]
    visited_sets = [torch.zeros(N, dtype=torch.bool, device=M.device) for _ in range(len(starts))]

    # Mark starting nodes as visited
    for i, start in enumerate(starts):
        visited_sets[i][start] = True

    neg_inf = torch.finfo(M.dtype).min

    # Build all paths step by step
    for _ in range(N - 1):  # N-1 steps to complete paths
        for path_idx in range(len(starts)):
            current_node = paths[path_idx][-1]  # Last node in current path
            visited = visited_sets[path_idx]

            # Find best next node for this path
            row = M[current_node].clone()
            row[visited] = neg_inf  # Mask out visited nodes
            next_node = int(torch.argmax(row).item())

            # Add to path and mark as visited
            paths[path_idx].append(next_node)
            visited_sets[path_idx][next_node] = True

    # Calculate greedy weights for all paths
    all_results: List[Tuple[List[int], float]] = []
    for path in paths:
        weight = path_weight_torch(M, path)
        all_results.append((path, weight))

    # Sort by greedy weight descending
    all_results.sort(key=lambda x: x[1], reverse=True)

    # Polishing: apply 2-opt to top-k (k may be 0)
    k = max(0, min(polish_top_k, len(all_results)))
    best_path, best_weight = None, float("-inf")

    # Consider polished versions of top-k
    for i in range(k):
        path_i = all_results[i][0]
        polished_path = two_opt_improve_torch(M, path_i, max_passes=2)
        polished_weight = path_weight_torch(M, polished_path)
        if polished_weight > best_weight:
            best_weight = polished_weight
            best_path = polished_path

    # If k == 0 or all polished were worse than some unpolished, also consider remaining greedy bests
    # (or use this as a fallback if best_path wasn't set)
    for path, weight in all_results:
        if weight > best_weight:
            best_weight = weight
            best_path = path

    return best_path, best_weight, all_results


@torch.no_grad()
def smart_interleaved_paths_torch(
    M: torch.Tensor,
    n_paths: int = 16,
    strategy: str = "mixed",
    polish_top_k: int = 1,
) -> Tuple[List[int], float]:
    """
    Interleaved path construction with smart starting point selection.

    Args:
        M: Similarity matrix
        n_paths: Number of paths to construct simultaneously
        strategy: How to choose starting points ("random", "endpoints", "diverse", "mixed")
        polish_top_k: How many top-scoring paths to polish with 2-opt (default 1)
    """
    N = M.size(0)
    n_paths = min(n_paths, N)

    if strategy == "random":
        starts = torch.randperm(N)[:n_paths].tolist()

    elif strategy == "endpoints":
        # Find likely video endpoints
        out_strength = M.sum(dim=1)
        in_strength = M.sum(dim=0)
        asymmetry = torch.abs(out_strength - in_strength)
        threshold = torch.quantile(M, 0.8)
        high_sim_count = (M > threshold).sum(dim=1).float()
        endpoint_score = asymmetry - high_sim_count
        starts = torch.topk(endpoint_score, n_paths).indices.tolist()

    elif strategy == "diverse":
        # Farthest-first selection for diverse starting points
        starts = []
        remaining = set(range(N))

        # Start with highest degree node
        first = int(torch.argmax(M.sum(dim=1)))
        starts.append(first)
        remaining.remove(first)

        # Add most diverse remaining nodes
        while len(starts) < n_paths and remaining:
            remaining_list = list(remaining)
            if starts:
                # choose the node with the lowest max similarity to the chosen set
                min_sims = M[remaining_list][:, starts].max(dim=1)[0]
                next_idx = int(torch.argmin(min_sims))
            else:
                next_idx = 0
            next_node = remaining_list[next_idx]
            starts.append(next_node)
            remaining.remove(next_node)

    elif strategy == "mixed":
        # Combine strategies
        third = max(1, n_paths // 3)

        # Random starts
        random_starts = torch.randperm(N)[:third].tolist()

        # Endpoint starts
        out_strength = M.sum(dim=1)
        in_strength = M.sum(dim=0)
        asymmetry = torch.abs(out_strength - in_strength)
        threshold = torch.quantile(M, 0.8)
        high_sim_count = (M > threshold).sum(dim=1).float()
        endpoint_score = asymmetry - high_sim_count
        endpoint_starts = torch.topk(endpoint_score, third).indices.tolist()

        # High degree starts
        node_strengths = M.sum(dim=1)
        degree_starts = torch.topk(node_strengths, n_paths - 2 * third).indices.tolist()

        # Dedup while preserving order
        seen = set()
        starts = []
        for s in (random_starts + endpoint_starts + degree_starts):
            if s not in seen:
                seen.add(int(s))
                starts.append(int(s))
        # If dedup shrank the list, backfill randomly
        if len(starts) < n_paths:
            remaining = [i for i in range(N) if i not in seen]
            if remaining:
                rand_perm = torch.tensor(remaining)[torch.randperm(len(remaining))].tolist()
                starts.extend(rand_perm[: n_paths - len(starts)])
    else:
        # Fallback to random if unknown strategy
        starts = torch.randperm(N)[:n_paths].tolist()

    best_path, best_weight, _ = interleaved_multi_path_torch(
        M,
        n_paths=len(starts),
        start_indices=starts,
        polish_top_k=polish_top_k,
        dedup_starts=True,
    )
    return best_path, best_weight


@torch.no_grad()
def video_recovery_interleaved_torch(
    M: torch.Tensor,
    n_paths: int = 16,
    try_reverse: bool = True,
    polish_top_k: int = 1,
) -> Tuple[List[int], float]:
    """
    Video sequence recovery using interleaved multi-path construction.
    Runs forward and (optionally) reverse (via M.T) and returns the better path.
    """
    # Forward direction
    path_fwd, weight_fwd = smart_interleaved_paths_torch(
        M, n_paths=n_paths, strategy="endpoints", polish_top_k=polish_top_k
    )

    if not try_reverse:
        return path_fwd, weight_fwd

    # Reverse direction
    path_rev, weight_rev = smart_interleaved_paths_torch(
        M.T, n_paths=n_paths, strategy="endpoints", polish_top_k=polish_top_k
    )
    path_rev.reverse()  # Correct order for reverse direction

    # Return better result
    if weight_fwd >= weight_rev:
        return path_fwd, weight_fwd
    else:
        return path_rev, weight_rev


# Helper functions (same as before)
@torch.no_grad()
def path_weight_torch(M: torch.Tensor, path: List[int]) -> float:
    """Calculate total weight of a path."""
    idx_from = torch.tensor(path[:-1], device=M.device)
    idx_to = torch.tensor(path[1:], device=M.device)
    return float(M[idx_from, idx_to].sum().item())


@torch.no_grad()
def two_opt_improve_torch(M: torch.Tensor, path: List[int], max_passes: int = 3) -> List[int]:
    """2-opt improvement for paths on a directed, weighted graph."""
    best = path[:]
    N = len(best)
    improved = True
    passes = 0

    while improved and passes < max_passes:
        improved = False
        passes += 1
        for i in range(1, N - 2):
            for k in range(i + 1, N - 1):
                a, b = best[i - 1], best[i]
                c, d = best[k], best[k + 1]
                w_before = M[a, b] + M[c, d]
                w_after = M[a, c] + M[b, d]
                if (w_after - w_before).item() > 1e-12:
                    best[i : k + 1] = reversed(best[i : k + 1])
                    improved = True
    return best


def most_similar_path_old(M, n_starts=1, polish_top_k=1):
    path, weight, all_paths = interleaved_multi_path_torch(M, n_paths=n_starts, polish_top_k=polish_top_k)
    return path, weight




# Longest Hamiltonian Path Toolkit (PyTorch-first; NumPy also accepted)
# --------------------------------------------------------------
# Usage:
#   best, results = solve_longest_hamiltonian_path(M)
#   # best["path"] -> list[int], best["weight"] -> float
#   # results -> dict per method
#
# Notes:
# - M can be torch.Tensor [n,n] or numpy.ndarray. Symmetry not required.
# - We interpret larger weights as "better." Diagonal is ignored.
# - All methods return a *path* (no return-to-start).
# - Defaults aim to run on ~1000 nodes within reasonable time. Tune as needed.


# --------------------------
# Utils
# --------------------------

def _to_tensor(M) -> torch.Tensor:
    if isinstance(M, torch.Tensor):
        T = M
    else:
        T = torch.tensor(M)
    if T.dtype not in (torch.float32, torch.float64):
        T = T.float()
    return T

def _prep_matrix(M: torch.Tensor) -> torch.Tensor:
    # Make a working copy, ignore self loops by setting diag to -inf
    M = M.clone()
    n = M.shape[0]
    idx = torch.arange(n)
    M[idx, idx] = float("-inf")
    return M

def path_weight(M: torch.Tensor, path: List[int]) -> float:
    if len(path) <= 1:
        return 0.0
    p = torch.tensor(path, dtype=torch.long)
    return float(M[p[:-1], p[1:]].sum().item())

def pair_with_max_edge(M: torch.Tensor) -> Tuple[int, int]:
    # Returns (i,j) with maximum M[i,j], i != j
    n = M.shape[0]
    # Mask diagonal is already -inf, so argmax over flattened works
    flat_idx = torch.argmax(M.view(-1)).item()
    i, j = divmod(flat_idx, n)
    return int(i), int(j)

def farthest_neighbor_path_greedy(M: torch.Tensor, start: Optional[int] = None) -> List[int]:
    # Simple fast greedy (two-ended extension) to get an initial path.
    n = M.shape[0]
    unvisited = set(range(n))
    if start is None:
        i, j = pair_with_max_edge(M)
    else:
        i = start
        # choose farthest neighbor from start
        j = torch.argmax(M[i]).item()
        if j == i:
            # fallback
            _, j = pair_with_max_edge(M)

    path = [i, j]
    unvisited.discard(i); unvisited.discard(j)

    left, right = i, j
    while unvisited:
        # best extension on left
        u_left = None
        if unvisited:
            cand_left = torch.tensor(list(unvisited), dtype=torch.long)
            gains_left = M[cand_left, left]
            k_left = torch.argmax(gains_left).item()
            u_left = int(cand_left[k_left])

        # best extension on right
        u_right = None
        if unvisited:
            cand_right = torch.tensor(list(unvisited), dtype=torch.long)
            gains_right = M[right, cand_right]
            k_right = torch.argmax(gains_right).item()
            u_right = int(cand_right[k_right])

        # pick the better side
        if u_left is None and u_right is None:
            break
        gain_left = M[u_left, left] if u_left is not None else float("-inf")
        gain_right = M[right, u_right] if u_right is not None else float("-inf")

        if gain_left >= gain_right:
            path.insert(0, u_left)
            unvisited.remove(u_left)
            left = u_left
        else:
            path.append(u_right)
            unvisited.remove(u_right)
            right = u_right
    return path

# --------------------------
# 1) Regret-k Insertion (Longest Path)
# --------------------------

def regret_insertion_longest_path(
    M_in,
    k: int = 2,
    seed: Optional[int] = None,
    init_pair: Optional[Tuple[int,int]] = None,
) -> List[int]:
    """
    Build a path by inserting nodes using a regret-k rule:
    - At each step, for each unvisited node, compute best and k-th best insertion gains.
    - Choose the node with largest "regret" = best - second_best (for k=2).
    Maximizes total sum of adjacent edge weights (longest path).
    """
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)

    M = _prep_matrix(_to_tensor(M_in))
    n = M.shape[0]

    # Seed path with the strongest edge (or provided pair)
    if init_pair is None:
        a, b = pair_with_max_edge(M)
    else:
        a, b = init_pair
    path = [a, b]

    remaining = torch.tensor([i for i in range(n) if i not in path], dtype=torch.long)

    while remaining.numel() > 0:
        m = len(path)
        P = torch.tensor(path, dtype=torch.long)

        # End insert gains
        gains_left  = M[remaining, P[0]]           # shape (r,)
        gains_right = M[P[-1], remaining]          # shape (r,)

        # Internal insert gains (between path[i] and path[i+1])
        if m > 1:
            A = P[:-1]  # (m-1,)
            B = P[1:]   # (m-1,)
            Ma = M[A][:, remaining]                # (m-1, r)
            Mb = M[remaining][:, B].T             # (m-1, r)
            Mab = M[A, B].unsqueeze(1)            # (m-1, 1)
            gains_between = Ma + Mb - Mab         # (m-1, r)
            gains_pos = torch.vstack([gains_left.unsqueeze(0), gains_between, gains_right.unsqueeze(0)])
        else:
            # degenerate, shouldn't happen since m>=2
            gains_pos = torch.vstack([gains_left.unsqueeze(0), gains_right.unsqueeze(0)])

        # top-2 positions for each node (col-wise topk)
        top_vals, top_idx = torch.topk(gains_pos, k=min(2, gains_pos.shape[0]), dim=0)
        if top_vals.shape[0] == 1:
            regrets = top_vals[0]  # only one position exists
            best_pos = top_idx[0]
        else:
            regrets = top_vals[0] - top_vals[1]
            best_pos = top_idx[0]

        # pick node with largest regret; on tie, pick with largest best gain
        best_node_idx = torch.argmax(torch.stack([regrets, top_vals[0]], dim=0), dim=1)[0]
        # Actually, the above stacks across axes; simpler:
        # prioritize regret, then best gain:
        max_regret = torch.max(regrets)
        mask = (regrets == max_regret)
        candidates = torch.nonzero(mask, as_tuple=True)[0]
        # break ties by best immediate gain
        if candidates.numel() > 1:
            cand_best = top_vals[0, candidates]
            winner_local = candidates[torch.argmax(cand_best)]
        else:
            winner_local = candidates[0]

        insert_pos = int(best_pos[winner_local].item())
        node = int(remaining[winner_local].item())

        # realize insertion
        if insert_pos == 0:
            path.insert(0, node)
        elif insert_pos == len(path):
            path.append(node)
        else:
            path.insert(insert_pos, node)

        # update remaining
        remaining = torch.tensor([i for i in remaining.tolist() if i != node], dtype=torch.long)

    return path

# --------------------------
# 2) Genetic Algorithm (Longest Path)
# --------------------------

def _eval_paths_batch(M: torch.Tensor, paths: List[List[int]]) -> torch.Tensor:
    # returns tensor of path weights (len(paths),)
    if not paths:
        return torch.empty(0)
    weights = []
    for p in paths:
        weights.append(path_weight(M, p))
    return torch.tensor(weights, dtype=torch.float32)

def _mutation_swap(path: List[int], rate: float, rng: random.Random):
    if rng.random() < rate:
        n = len(path)
        i, j = rng.randrange(n), rng.randrange(n)
        path[i], path[j] = path[j], path[i]

def _order_crossover(p1: List[int], p2: List[int], rng: random.Random) -> List[int]:
    n = len(p1)
    i, j = sorted((rng.randrange(n), rng.randrange(n)))
    child = [None] * n
    # inherit slice from p1
    child[i:j+1] = p1[i:j+1]
    used = set(child[i:j+1])
    # fill from p2
    idx = (j + 1) % n
    for x in p2:
        if x in used:
            continue
        while child[idx] is not None:
            idx = (idx + 1) % n
        child[idx] = x
    return child

def genetic_algorithm_longest_path(
    M_in,
    pop_size: int = 30,
    generations: int = 120,
    mutation_rate: float = 0.25,
    elite: int = 2,
    tournament_k: int = 3,
    seed: Optional[int] = None,
) -> List[int]:
    """
    GA with Order Crossover (OX) and swap mutation. Keeps elites.
    Tuned for scalability; keep pop/gens modest for n~1000.
    """
    rng = random.Random(seed)
    if seed is not None:
        torch.manual_seed(seed)

    M = _prep_matrix(_to_tensor(M_in))
    n = M.shape[0]

    # Initialize population using greedy seeds + random perms for diversity
    population: List[List[int]] = []
    # A couple of greedy starts
    for s in range(min(4, pop_size)):
        start = rng.randrange(n)
        population.append(farthest_neighbor_path_greedy(M, start=start))
    # Fill the rest randomly
    while len(population) < pop_size:
        perm = list(range(n))
        rng.shuffle(perm)
        population.append(perm)

    fitness = _eval_paths_batch(M, population)

    def tournament_select() -> List[int]:
        idxs = [rng.randrange(pop_size) for _ in range(tournament_k)]
        best = max(idxs, key=lambda i: fitness[i].item())
        return population[best][:]

    for _ in range(generations):
        # Elites
        elite_idx = torch.topk(fitness, k=elite).indices.tolist()
        new_pop: List[List[int]] = [population[i][:] for i in elite_idx]

        # Children
        while len(new_pop) < pop_size:
            p1 = tournament_select()
            p2 = tournament_select()
            child = _order_crossover(p1, p2, rng)
            _mutation_swap(child, mutation_rate, rng)
            new_pop.append(child)

        population = new_pop
        fitness = _eval_paths_batch(M, population)

    best_idx = int(torch.argmax(fitness).item())
    return population[best_idx][:]

# --------------------------
# 3) Ant Colony Optimization (Longest Path)
# --------------------------

def ant_colony_longest_path(
    M_in,
    ants: int = 16,
    iterations: int = 60,
    alpha: float = 1.0,   # pheromone influence
    beta: float = 3.0,    # heuristic influence
    rho: float = 0.6,     # evaporation rate
    q: float = 1.0,       # deposit factor
    candidate_k: int = 30,
    seed: Optional[int] = None,
) -> List[int]:
    """
    ACO adapted for longest Hamiltonian path:
    - Heuristic (eta) favors large weights.
    - Candidate lists (top-k per node) limit branching.
    - Global-best pheromone update each iteration.
    """
    rng = random.Random(seed)
    if seed is not None:
        torch.manual_seed(seed)

    M = _prep_matrix(_to_tensor(M_in))
    n = M.shape[0]

    # Heuristic: shift to positive
    m_min = torch.nan_to_num(torch.min(M[M != float("-inf")]), nan=0.0)
    eps = 1e-9
    eta = (M - m_min + eps)  # larger is better

    # Candidate lists: top-k neighbors per node
    k = min(candidate_k, max(1, n - 1))
    top_vals, top_idx = torch.topk(eta, k=k, dim=1)  # per row

    # Pheromone init
    tau = torch.ones_like(M) * (1.0 / (n * n))

    def build_path() -> List[int]:
        start = rng.randrange(n)
        path = [start]
        visited = set([start])
        cur = start

        for _ in range(n - 1):
            cand = [int(j) for j in top_idx[cur].tolist() if j not in visited]
            if not cand:
                # fallback: pick best among all unvisited
                remaining = list(set(range(n)) - visited)
                if not remaining:
                    break
                candidates = torch.tensor(remaining, dtype=torch.long)
                desir = tau[cur, candidates].pow(alpha) * eta[cur, candidates].pow(beta)
                if torch.all(desir <= 0):
                    nxt = int(candidates[torch.argmax(eta[cur, candidates])].item())
                else:
                    probs = desir / torch.clamp(desir.sum(), min=eps)
                    nxt = int(candidates[torch.multinomial(probs, 1).item()])
            else:
                cands = torch.tensor(cand, dtype=torch.long)
                desir = tau[cur, cands].pow(alpha) * eta[cur, cands].pow(beta)
                if torch.all(desir <= 0):
                    nxt = int(cands[torch.argmax(eta[cur, cands])].item())
                else:
                    probs = desir / torch.clamp(desir.sum(), min=eps)
                    nxt = int(cands[torch.multinomial(probs, 1).item()])
            path.append(nxt)
            visited.add(nxt)
            cur = nxt
        return path

    best_path = None
    best_w = float("-inf")

    for _ in range(iterations):
        paths = [build_path() for _ in range(ants)]
        weights = [_eval_paths_batch(M, [p])[0].item() for p in paths]

        # iteration-best
        it_best_idx = max(range(len(paths)), key=lambda i: weights[i])
        it_path = paths[it_best_idx]
        it_w = weights[it_best_idx]
        if it_w > best_w:
            best_w, best_path = it_w, it_path

        # Evaporate
        tau *= (1.0 - rho)

        # Deposit along iteration-best (or global-best—choose either; here: iteration-best)
        deposit = q * max(it_w, 0.0) / (n - 1 + eps)
        for u, v in zip(it_path[:-1], it_path[1:]):
            tau[u, v] += deposit
            tau[v, u] += deposit  # if symmetric

        # Optional: cap pheromone to avoid explosion
        tau.clamp_(min=1e-12, max=1e6)

    return best_path if best_path is not None else list(range(n))

# --------------------------
# 4) Beam Search (candidate-guided, one-end growth)
# --------------------------

def beam_search_longest_path(
    M_in,
    beam_size: int = 10,
    candidate_k: int = 30,
    restarts: int = 3,
    seed: Optional[int] = None,
) -> List[int]:
    """
    Bounded beam search growing paths from strong starts, expanding only top-k neighbors.
    Keeps 'beam_size' partial paths at each depth.
    """
    rng = random.Random(seed)
    if seed is not None:
        torch.manual_seed(seed)

    M = _prep_matrix(_to_tensor(M_in))
    n = M.shape[0]
    k = min(candidate_k, max(1, n - 1))

    # Candidate lists
    _, top_idx = torch.topk(M, k=k, dim=1)

    def run_once(start: int) -> List[int]:
        # Each beam item: (path:list, last:int, weight:float, visited:set)
        beams = [([start], start, 0.0, {start})]
        for _ in range(n - 1):
            expanded = []
            for path, last, w, visited in beams:
                cands = [int(j) for j in top_idx[last].tolist() if j not in visited]
                if not cands:
                    # fallback: global best unvisited
                    remaining = list(set(range(n)) - visited)
                    if not remaining:
                        expanded.append((path, last, w, visited))
                        continue
                    best_j = max(remaining, key=lambda j: float(M[last, j].item()))
                    expanded.append((path + [best_j], best_j, w + float(M[last, best_j].item()), visited | {best_j}))
                    continue

                # Generate children for top candidates
                for j in cands:
                    expanded.append((path + [j], j, w + float(M[last, j].item()), visited | {j}))

            # Prune to beam_size by partial weight
            if len(expanded) > beam_size:
                expanded.sort(key=lambda x: x[2], reverse=True)
                beams = expanded[:beam_size]
            else:
                beams = expanded

        # Return best completed path
        beams.sort(key=lambda x: x[2], reverse=True)
        return beams[0][0]

    best_path = None
    best_w = float("-inf")

    # Pick diverse strong starts: nodes with largest row-sum
    row_sums = torch.nan_to_num(M, neginf=0.0).clamp_min(0).sum(dim=1)
    top_starts = torch.topk(row_sums, k=min(restarts, n)).indices.tolist()
    # add a couple random starts for diversity
    while len(top_starts) < restarts and len(top_starts) < n:
        r = rng.randrange(n)
        if r not in top_starts:
            top_starts.append(r)

    for s in top_starts:
        path = run_once(s)
        w = path_weight(M, path)
        if w > best_w:
            best_w, best_path = w, path

    return best_path

# --------------------------
# 5) Iterated Greedy (destroy & regret-repair, SA-like acceptance)
# --------------------------

def iterated_greedy_longest_path(
    M_in,
    iters: int = 60,
    destroy_frac: float = 0.05,
    regret_k: int = 2,
    seed: Optional[int] = None,
    start_temp: float = 0.01,
    cooling: float = 0.98,
) -> List[int]:
    """
    Start with a greedy path; repeatedly remove a block and reinsert nodes via regret-k.
    Accept improvements or (rarely) worse moves with exp((new-old)/T).
    """
    rng = random.Random(seed)
    if seed is not None:
        torch.manual_seed(seed)

    M = _prep_matrix(_to_tensor(M_in))
    n = M.shape[0]

    cur = farthest_neighbor_path_greedy(M)
    cur_w = path_weight(M, cur)

    T = start_temp

    def regret_insert_sequence(path: List[int], nodes: List[int]) -> List[int]:
        # Reuse regret insertion logic, inserting given nodes into existing path
        P = path[:]
        remaining = torch.tensor(nodes, dtype=torch.long)
        while remaining.numel() > 0:
            m = len(P)
            P_t = torch.tensor(P, dtype=torch.long)

            gains_left  = M[remaining, P_t[0]]
            gains_right = M[P_t[-1], remaining]
            if m > 1:
                A = P_t[:-1]; B = P_t[1:]
                Ma = M[A][:, remaining]
                Mb = M[remaining][:, B].T
                Mab = M[A, B].unsqueeze(1)
                gains_between = Ma + Mb - Mab
                gains_pos = torch.vstack([gains_left.unsqueeze(0), gains_between, gains_right.unsqueeze(0)])
            else:
                gains_pos = torch.vstack([gains_left.unsqueeze(0), gains_right.unsqueeze(0)])

            top_vals, top_idx = torch.topk(gains_pos, k=min(2, gains_pos.shape[0]), dim=0)
            if top_vals.shape[0] == 1:
                regrets = top_vals[0]; best_pos = top_idx[0]
            else:
                regrets = top_vals[0] - top_vals[1]; best_pos = top_idx[0]

            max_regret = torch.max(regrets)
            mask = (regrets == max_regret)
            candidates = torch.nonzero(mask, as_tuple=True)[0]
            if candidates.numel() > 1:
                cand_best = top_vals[0, candidates]
                winner_local = candidates[torch.argmax(cand_best)]
            else:
                winner_local = candidates[0]

            pos = int(best_pos[winner_local].item())
            node = int(remaining[winner_local].item())

            if pos == 0:
                P.insert(0, node)
            elif pos == len(P):
                P.append(node)
            else:
                P.insert(pos, node)

            remaining = torch.tensor([i for i in remaining.tolist() if i != node], dtype=torch.long)
        return P

    for _ in range(iters):
        # Destroy: remove a contiguous block of d nodes
        d = max(1, int(round(destroy_frac * n)))
        if d >= n:
            break
        s = rng.randrange(0, n - d + 1)
        removed = cur[s:s+d]
        kept = cur[:s] + cur[s+d:]

        # Repair: regret-insert removed nodes
        new_path = regret_insert_sequence(kept, removed)
        new_w = path_weight(M, new_path)

        # Accept if better or probabilistically worse
        if new_w >= cur_w:
            cur, cur_w = new_path, new_w
        else:
            if rng.random() < math.exp((new_w - cur_w) / max(T, 1e-12)):
                cur, cur_w = new_path, new_w

        T *= cooling

    return cur

# --------------------------
# Meta Runner
# --------------------------

def solve_longest_hamiltonian_path(
    M_in,
    # Per-method tunables (override as needed)
    regret_params: dict = None,
    ga_params: dict = None,
    aco_params: dict = None,
    beam_params: dict = None,
    ig_params: dict = None,
    methods: Tuple[str, ...] = ("regret", "ga", "aco", "beam", "ig"),
    seed: Optional[int] = 42,
) -> Tuple[Dict, Dict[str, Dict]]:
    """
    Runs selected methods, returns:
      best: {"method": str, "path": List[int], "weight": float}
      results: {method_name: {"path": List[int], "weight": float, "params": dict}}
    """
    regret_params = regret_params or {}
    ga_params = ga_params or {}
    aco_params = aco_params or {}
    beam_params = beam_params or {}
    ig_params = ig_params or {}

    # unify seed propagation without forcing it
    if seed is not None:
        for D in (regret_params, ga_params, aco_params, beam_params, ig_params):
            D.setdefault("seed", seed)

    M = _to_tensor(M_in)
    results: Dict[str, Dict] = {}
    best = {"method": None, "path": None, "weight": float("-inf")}

    def consider(name: str, path: List[int], params: dict):
        w = path_weight(M, path)
        results[name] = {"path": path, "weight": w, "params": params.copy()}
        nonlocal best
        if w > best["weight"]:
            best = {"method": name, "path": path, "weight": w}

    if "regret" in methods:
        p = regret_insertion_longest_path(M, **regret_params)
        consider("regret", p, regret_params)

    if "ga" in methods:
        p = genetic_algorithm_longest_path(M, **ga_params)
        consider("ga", p, ga_params)

    # if "aco" in methods:
    #     p = ant_colony_longest_path(M, **aco_params)
    #     consider("aco", p, aco_params)

    # if "beam" in methods:
    #     p = beam_search_longest_path(M, **beam_params)
    #     consider("beam", p, beam_params)

    if "ig" in methods:
        p = iterated_greedy_longest_path(M, **ig_params)
        consider("ig", p, ig_params)

    return best, results


def most_similar_path(M):
    res = solve_longest_hamiltonian_path(M)
    path = res[0]['path']
    return path