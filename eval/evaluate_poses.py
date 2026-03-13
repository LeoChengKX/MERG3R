from evo.core.trajectory import PosePath3D, PoseTrajectory3D
import evo.core.lie_algebra as lie
import numpy as np
import json
from evo_utils import eval_metrics
import pdb
import os
import argparse

parser = argparse.ArgumentParser(description="Process command-line arguments.")
parser.add_argument("--tgt", type=str, default=None)
parser.add_argument("--pred", type=str, default=None)
parser.add_argument("--path", type=str, default=None)

# Pose matrix storage format
parser.add_argument("--pred_format", choices=["c2w","w2c"], default="c2w")
parser.add_argument("--tgt_format",  choices=["c2w","w2c"], default="c2w")

# NEW: camera-frame conventions for each input and a target convention
parser.add_argument("--pred_coords",  choices=["opencv","opengl"], default="opencv",
                    help="Camera coordinate convention used by PRED matrices.")
parser.add_argument("--tgt_coords",   choices=["opencv","opengl"], default="opencv",
                    help="Camera coordinate convention used by TGT matrices.")
parser.add_argument("--target_coords", choices=["opencv","opengl"], default="opengl",
                    help="Convention to convert BOTH streams into before evaluation.")

args = parser.parse_args()

def tranforms_to_trajectory(transforms):
    frames = sorted(transforms["frames"], key = lambda x: x["file_path"])
    frames = np.stack([x["transform_matrix"] for x in frames], 0)
    return PosePath3D(poses_se3=frames)

def read_json(json_path):
    with open(json_path, "r") as f:
        return json.load(f)

def check_so3(rot):
    return True

def visualize_trajectory(traj, traj_gt, save_path=None):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    pred_pos = traj.positions_xyz
    gt_pos = traj_gt.positions_xyz

    ax.plot(pred_pos[:, 0], pred_pos[:, 1], pred_pos[:, 2], 'r-', label='Predicted')
    ax.plot(gt_pos[:, 0], gt_pos[:, 1], gt_pos[:, 2], 'b-', label='Ground Truth')

    ax.plot([pred_pos[0, 0]], [pred_pos[0, 1]], [pred_pos[0, 2]], 'ro', markersize=8)
    ax.plot([pred_pos[-1, 0]], [pred_pos[-1, 1]], [pred_pos[-1, 2]], 'rx', markersize=8)
    ax.plot([gt_pos[0, 0]], [gt_pos[0, 1]], [gt_pos[0, 2]], 'bo', markersize=8)
    ax.plot([gt_pos[-1, 0]], [gt_pos[-1, 1]], [gt_pos[-1, 2]], 'bx', markersize=8)

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('Camera Trajectory Comparison')

    def _range(P):
        return max(P[:,0].ptp(), P[:,1].ptp(), P[:,2].ptp())

    max_range = max(_range(pred_pos), _range(gt_pos))
    mid = (np.vstack([pred_pos, gt_pos]).max(0) + np.vstack([pred_pos, gt_pos]).min(0)) * 0.5
    ax.set_xlim(mid[0] - max_range/2, mid[0] + max_range/2)
    ax.set_ylim(mid[1] - max_range/2, mid[1] + max_range/2)
    ax.set_zlim(mid[2] - max_range/2, mid[2] + max_range/2)

    plt.legend()
    if save_path:
        viz_dir = os.path.dirname(save_path)
        if viz_dir and not os.path.exists(viz_dir):
            os.makedirs(viz_dir)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Trajectory visualization saved to {save_path}")

def filter_gt_by_target(gt, target):
    target_filenames = set([os.path.basename(x["file_path"]).split('.')[0] for x in target["frames"]])
    gt_filtered = [x for x in gt["frames"] if os.path.basename(x["file_path"]).split('.')[0] in target_filenames]
    gt["frames"] = gt_filtered
    return gt

def filter_gt_by_seq_num(frames_pred, frames_gt):
    filtered_gt = []
    pred_path_example = (frames_pred[0]['file_path']).split('/')
    gt_path_example = (frames_gt[0]['file_path']).split('/')
    pred_have_seq = False
    for i in pred_path_example:
        if 'seq' in i:
            target_seq_num = i
            pred_have_seq = True
            break
    gt_have_seq = False
    for i in range(len(gt_path_example)):
        if 'seq' in gt_path_example[i]:
            gt_path_seq_idx = i
            gt_have_seq = True
            break
    for frame in frames_gt:
        if gt_have_seq and pred_have_seq:
            gt_seq_num = ((frame['file_path']).split('/'))[gt_path_seq_idx]
            if gt_seq_num == target_seq_num:
                filtered_gt.append(frame)
        else:
            filtered_gt.append(frame)
    return filtered_gt

# ---------- NEW: convention conversion helpers ----------
_F = np.diag([1.0, -1.0, -1.0, 1.0])  # flips Y,Z (OpenCV <-> OpenGL)

def convert_coords(mat4, fmt, src_coords, dst_coords):
    """
    Convert a single 4x4 pose matrix between camera-frame conventions.
    fmt: 'c2w' or 'w2c' for the CURRENT matrix
    src_coords, dst_coords: 'opencv' or 'opengl'
    """
    if src_coords == dst_coords:
        return mat4
    if fmt == "c2w":
        # camera-to-world: change of camera basis is applied on the RIGHT
        return mat4 @ _F
    else:  # 'w2c'
        # world-to-camera: change of camera basis is applied on the LEFT
        return _F @ mat4

def to_target_fmt(mat4, src_fmt, dst_fmt):
    if src_fmt == dst_fmt:
        return mat4
    return np.linalg.inv(mat4)

def harmonize_frames(frames, src_fmt, src_coords, target_fmt="c2w", target_coords="opengl"):
    """
    For a list of frame dicts with 'transform_matrix', return a stacked array of
    matrices converted to the desired format and coordinates.
    """
    mats = []
    for f in frames:
        M = np.array(f["transform_matrix"], dtype=np.float64)
        # 1) bring to desired FORMAT (c2w/w2c)
        M = to_target_fmt(M, src_fmt, target_fmt)
        # 2) bring to desired COORDS (opencv/opengl)
        M = convert_coords(M, target_fmt, src_coords, target_coords)
        mats.append(M)
    return np.stack(mats, 0)

def load_stack(json_path):
    data = read_json(json_path)
    frames = sorted(data["frames"], key=lambda x: x["file_path"])
    return data, frames

def evaluate_poses(transforms_pred_path, transforms_gt_path, path = "./"):
    transforms_pred = read_json(transforms_pred_path)
    transforms_gt   = read_json(transforms_gt_path)

    get_frames = lambda x: sorted(x["frames"], key = lambda x: x["file_path"])
    frames_pred = get_frames(transforms_pred)
    frames_gt   = get_frames(transforms_gt)
    frames_gt   = filter_gt_by_seq_num(frames_pred, frames_gt)

    # Try to match by filenames if counts differ
    poses_pred_list, poses_gt_list = [], []
    if len(frames_gt) != len(frames_pred):
        print("Number of frames differ; sampling GT to match PRED by filenames...")
        assert len(frames_gt) > len(frames_pred), "Ground truth has fewer frames than prediction"
        p_idx = 0
        for i in range(len(frames_gt)):
            fname_pred = os.path.basename(frames_pred[p_idx]["file_path"]).split('.')[0]
            fname_gt   = os.path.basename(frames_gt[i]["file_path"]).split('.')[0]
            if p_idx < len(frames_pred) and fname_pred == fname_gt:
                pose_pred = np.array(frames_pred[p_idx]['transform_matrix'])
                if check_so3(pose_pred[:3, :3]):
                    poses_pred_list.append(frames_pred[p_idx])
                    poses_gt_list.append(frames_gt[i])
                else:
                    print("Bad SO3 in prediction, skipping:", fname_pred)
                p_idx += 1
            if p_idx == len(frames_pred):
                break
    else:
        poses_pred_list = frames_pred
        poses_gt_list   = frames_gt

    print("Matched frames:", len(poses_gt_list), len(poses_pred_list))
    assert len(poses_gt_list) == len(poses_pred_list), "Frame counts still differ after matching."

    # ---------- Harmonize BOTH streams to a common (format, coords) ----------
    target_fmt    = "c2w"
    target_coords = args.target_coords  # 'opengl' (default) or 'opencv'

    poses_pred = harmonize_frames(
        poses_pred_list, src_fmt=args.pred_format, src_coords=args.pred_coords,
        target_fmt=target_fmt, target_coords=target_coords
    )
    poses_gt = harmonize_frames(
        poses_gt_list, src_fmt=args.tgt_format, src_coords=args.tgt_coords,
        target_fmt=target_fmt, target_coords=target_coords
    )

    traj    = PosePath3D(poses_se3=poses_pred)
    traj_gt = PosePath3D(poses_se3=poses_gt)

    eval_metrics(traj, traj_gt, filename=path)
    visualize_trajectory(traj, traj_gt, save_path=path.replace(".txt", "_traj.png"))

if __name__=="__main__":
    args = parser.parse_args()

    tgt  = args.tgt
    pred = args.pred
    path = args.path if args.path is not None else os.path.join(os.path.dirname(pred), "results.txt")

    evaluate_poses(pred, tgt, path)
