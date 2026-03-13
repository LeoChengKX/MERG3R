#!/usr/bin/env python3
import argparse, json, os, math
from itertools import combinations
import numpy as np

# ----------------- Utils -----------------

def normalize_stem(file_path: str) -> str:
    """
    Robust 'stem' for NeRF file_path. Handles entries like:
      'images/0001.png', './train/r_0', or 'r_0.png'
    """
    base = os.path.basename(str(file_path))
    stem, ext = os.path.splitext(base)
    return stem if stem else base  # if no ext, use base as-is

def geodesic_angle_deg(Ra: np.ndarray, Rb: np.ndarray) -> float:
    """Angle between rotations Ra and Rb (degrees)."""
    R = Ra.T @ Rb
    tr = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(tr)))

def angle_between_dirs_deg(a: np.ndarray, b: np.ndarray, ignore_sign: bool=False) -> float:
    """Angle between 3D directions (degrees). If ignore_sign, use |dot|."""
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 180.0
    a = a / na; b = b / nb
    dot = float(np.dot(a, b))
    if ignore_sign: dot = abs(dot)
    dot = np.clip(dot, -1.0, 1.0)
    return float(np.degrees(np.arccos(dot)))

def w2c_to_c2w_one(R, t):
    Rc2w = R.T
    tc2w = -(Rc2w @ t)
    return Rc2w, tc2w

# ----------------- Readers -----------------

def read_nerf_json_as_dict_and_list(path: str):
    """
    Returns:
      dict_map: stem -> (R_c2w, t_c2w)
      ordered:  list of (stem, R_c2w, t_c2w) in file order
    """
    with open(path, 'r') as f:
        data = json.load(f)
    frames = data.get("frames", [])
    dict_map = {}
    ordered = []
    for fr in frames:
        stem = normalize_stem(fr.get("file_path", ""))
        T = np.array(fr["transform_matrix"], dtype=np.float64).reshape(4, 4)
        R = T[:3, :3]
        t = T[:3, 3]
        dict_map[stem] = (R, t)
        ordered.append((stem, R, t))
    return dict_map, ordered

# ----------------- Pairwise metrics -----------------

def relative_pose_c2c(Ri, ti, Rj, tj):
    """
    Absolute c2w poses (Ri,ti), (Rj,tj) -> relative from i to j, expressed in j:
      R_ij = R_j^T R_i
      t_ij = R_j^T (t_i - t_j)   (direction only for evaluation)
    """
    Rij = Rj.T @ Ri
    tij = Rj.T @ (ti - tj)
    return Rij, tij

def evaluate_pairs(R_pred, t_pred, R_gt, t_gt, pairs, tau_deg=30.0, auc_max_degree=30.0, ignore_sign=False):
    rot_errs, trans_errs = [], []
    for i, j in pairs:
        Rp_ij, tp_ij = relative_pose_c2c(R_pred[i], t_pred[i], R_pred[j], t_pred[j])
        Rg_ij, tg_ij = relative_pose_c2c(R_gt[i],   t_gt[i],   R_gt[j],   t_gt[j])
        rot_errs.append(geodesic_angle_deg(Rp_ij, Rg_ij))
        trans_errs.append(angle_between_dirs_deg(tp_ij, tg_ij, ignore_sign=ignore_sign))

    rot_errs = np.asarray(rot_errs, np.float64)
    trans_errs = np.asarray(trans_errs, np.float64)

    rra = float((rot_errs <= tau_deg).mean() * 100.0)
    rta = float((trans_errs <= tau_deg).mean() * 100.0)

    K = float(auc_max_degree)
    taus = np.arange(1, 31, 1, dtype=np.float64)
    rra_curve = np.array([(rot_errs <= t).mean() for t in taus], dtype=np.float64)
    rta_curve = np.array([(trans_errs <= t).mean() for t in taus], dtype=np.float64)
    acc_curve = np.minimum(rra_curve, rta_curve)
    # maa = float(acc_curve.mean() * 100.0)  # mAA@30
    auc = float(np.trapz(acc_curve, taus) / K * 100.0)

    return rra, rta, auc, taus, rra_curve, rta_curve, acc_curve

def w2c_to_c2w(R, T):
    # Convert w2c -> c2w
    R_c2w = np.transpose(R, (0, 2, 1))                       # (M, 3, 3)
    t_c2w = -(R_c2w @ T[..., None])[..., 0]                  # (M, 3)

    return R_c2w, t_c2w


S_OPENCV_TO_OPENGL = np.diag([1.0, -1.0, -1.0])

def adapt_coords(R, t, from_conv: str, to_conv: str):
    """Flip Y,Z axes if converting between OpenCV <-> OpenGL."""
    if from_conv == to_conv:
        return R, t
    S = S_OPENCV_TO_OPENGL
    # (c2w)  Xw = R * Xc + t ; if Xc' = S Xc, then R' = R S, t' = t
    return R @ S, t


# ----------------- Main -----------------

def main():
    ap = argparse.ArgumentParser(
        description="Evaluate RRA@30, RTA@30, mAA@30 between two NeRF transforms.json files."
    )
    ap.add_argument("--pred", required=True, help="Predicted poses transforms.json (c2w)")
    ap.add_argument("--tgt",   required=True, help="Ground-truth poses transforms.json (c2w)")
    ap.add_argument("--path", required=True, help="Path to write the output file to")
    ap.add_argument("--tau", type=float, default=30.0, help="Angle threshold (deg) for RRA/RTA (default 30).")
    ap.add_argument("--ignore_sign", action="store_true",
                    help="If set, translation is sign-agnostic (use |dot|).")
    ap.add_argument("--match", choices=["stem", "index"], default="stem",
                    help="How to match frames across files: by filename stem (default) or by index order.")
    ap.add_argument("--max_pairs", type=int, default=0,
                    help="If >0, randomly subsample this many pairs for evaluation.")
    ap.add_argument("--pred_conv", choices=["opengl", "opencv"], default="opengl",
                help="Coordinate convention for predicted poses (default opengl).")
    ap.add_argument("--gt_conv", choices=["opengl", "opencv"], default="opengl",
                    help="Coordinate convention for ground-truth poses (default opengl).")
    args = ap.parse_args()

    pred_map, pred_list = read_nerf_json_as_dict_and_list(args.pred)
    gt_map,   gt_list   = read_nerf_json_as_dict_and_list(args.tgt)

    # --- Apply coordinate convention conversion ---
    for k in pred_map:
        R, t = pred_map[k]
        pred_map[k] = adapt_coords(R, t, args.pred_conv, "opengl")

    for k in gt_map:
        R, t = gt_map[k]
        gt_map[k] = adapt_coords(R, t, args.gt_conv, "opengl")


    if args.match == "stem":
        # Intersect by filename stem
        common = sorted(set(pred_map.keys()) & set(gt_map.keys()))
        if len(common) < 2:
            raise RuntimeError(f"Need ≥2 common stems. Found {len(common)}.\n"
                               f"Example pred stems: {list(pred_map)[:5]}\n"
                               f"Example gt stems:   {list(gt_map)[:5]}")
        R_pred, t_pred, R_gt, t_gt = [], [], [], []
        for k in common:
            Rp, tp = pred_map[k]
            Rg, tg = gt_map[k]
            R_pred.append(Rp); t_pred.append(tp)
            R_gt.append(Rg);   t_gt.append(tg)
    else:  # index
        M = min(len(pred_list), len(gt_list))
        if M < 2:
            raise RuntimeError(f"Need ≥2 frames by index; got pred={len(pred_list)}, gt={len(gt_list)}.")
        R_pred = [pred_list[i][1] for i in range(M)]
        t_pred = [pred_list[i][2] for i in range(M)]
        R_gt   = [gt_list[i][1]   for i in range(M)]
        t_gt   = [gt_list[i][2]   for i in range(M)]
        common = [pred_list[i][0] for i in range(M)]  # stems from pred (for info)

    R_pred = np.stack(R_pred, axis=0)  # (M, 3, 3)
    t_pred = np.stack(t_pred, axis=0)  # (M, 3)
    R_gt   = np.stack(R_gt,   axis=0)  # (M, 3, 3)
    t_gt   = np.stack(t_gt,   axis=0)  # (M, 3)

    # R_pred, t_pred = w2c_to_c2w(R_pred, t_pred)

    M = len(common)
    idx_pairs = list(combinations(range(M), 2))
    if args.max_pairs and args.max_pairs < len(idx_pairs):
        rng = np.random.default_rng(0)
        sel = rng.choice(len(idx_pairs), size=args.max_pairs, replace=False)
        idx_pairs = [idx_pairs[i] for i in sel]

    rra, rta, auc, taus, rra_curve, rta_curve, acc_curve = evaluate_pairs(
        R_pred, t_pred, R_gt, t_gt, idx_pairs, tau_deg=args.tau, ignore_sign=args.ignore_sign
    )

    print(f"# Frames matched: {M}  (mode: {args.match})")
    print(f"# Pairs evaluated: {len(idx_pairs)}")
    print(f"RRA@{int(args.tau)}: {rra:.2f}%")
    print(f"RTA@{int(args.tau)}: {rta:.2f}%")
    print(f"AUC@30: {auc:.2f}%   (AUC of min(RRA,RTA) over 1°..30°)")

    # Optional pretty table (if pandas available)
    try:
        import pandas as pd
        df = pd.DataFrame({
            "tau_deg": taus,
            "RRA_%": rra_curve * 100.0,
            "RTA_%": rta_curve * 100.0,
            "AUC_%": acc_curve * 100.0
        })

        # print(df.to_string(index=False, float_format=lambda x: f"{x:6.2f}"))

        # Save the formatted DataFrame as text to the file
        with open(args.path, "w") as f:
            f.write(df.to_string(index=False, float_format=lambda x: f"{x:6.2f}"))
            f.write("\n\n")
            f.write(f"AUC@30: {auc:.2f}%\n")

        print(f"Results written to {args.path}")

    except Exception:
        pass

if __name__ == "__main__":
    main()