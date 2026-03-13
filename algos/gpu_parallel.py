import os
import time
import gc
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.multiprocessing as mp
import numpy as np

from vggt.models.vggt import VGGT
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map
from pi3.models.pi3 import Pi3
from .utils import remove_homogeneous_row, estimate_intrinsics_and_depth, compute_depth


# ------------------------------------------------------------
# Picklable model factory spec (for multiprocessing spawn).
# ------------------------------------------------------------
class ModelFactorySpec:
    """Picklable spec: worker builds model via _build_model_from_spec(spec)."""
    __slots__ = ("ckpt_path", "model_type")

    def __init__(self, ckpt_path: str, model_type: str = "vggt"):
        self.ckpt_path = ckpt_path
        self.model_type = model_type


def _build_model_from_spec(spec: "ModelFactorySpec") -> torch.nn.Module:
    """Top-level builder so it can be used in spawned processes."""
    if spec.model_type == "vggt":
        if os.path.isfile(spec.ckpt_path):
            model = make_vggt()
            state = torch.load(spec.ckpt_path, map_location="cpu")
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            model.load_state_dict(state, strict=True)
            return model
        else:
            return VGGT.from_pretrained(spec.ckpt_path)
    raise ValueError(f"Unknown model_type: {spec.model_type}")


# ------------------------------------------------------------
# Worker: runs inference on one GPU for a shard of indices.
# ------------------------------------------------------------
def _inference_worker(
    rank: int,
    world_size: int,
    make_model,  # Callable[[], torch.nn.Module] or ModelFactorySpec
    batches: Sequence[torch.Tensor],   # images on CPU (or pinned CPU)
    indices: Sequence[int],            # which items this rank should process
    size_hw: Tuple[int, int],
    need_features: bool,
    out_queue: mp.Queue,
    amp_dtype: torch.dtype = torch.bfloat16,
):
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    # Create a local model replica on this GPU (support both callable and picklable spec)
    if isinstance(make_model, ModelFactorySpec):
        model = _build_model_from_spec(make_model)
    else:
        model = make_model()
    model.eval()
    model.to(device)

    # Speed / memory
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Do NOT spam empty_cache() inside the loop; it usually hurts perf.
    # If you have fragmentation issues, call it once at the end.

    with torch.inference_mode():
        for idx in indices:
            images_cpu = batches[idx]
            prediction: Dict[str, Any] = {}

            # ---- forward pass on GPU ----
            with torch.amp.autocast('cuda', dtype=amp_dtype):
                images = images_cpu[None].to(device, non_blocking=True)  # (1, C, H, W) or whatever your model expects

                if isinstance(model, VGGT):
                    aggregated_tokens_list, ps_idx, patch_tokens = model.aggregator(images)

                    pose_enc = model.camera_head(aggregated_tokens_list)[-1]
                    depth_map, depth_conf = model.depth_head(aggregated_tokens_list, images, ps_idx)

                    if need_features:
                        feature_maps = model.track_head.feature_extractor(
                            aggregated_tokens_list, images, ps_idx
                        )

                elif isinstance(model, Pi3):
                    res = model(images)

                else:
                    raise TypeError(f"Unsupported model type: {type(model)}")

            # ---- postprocess + move outputs to CPU (still inside inference_mode) ----
            if isinstance(model, VGGT):
                prediction["depth"] = depth_map.to(dtype=torch.float32).cpu()
                prediction["depth_conf"] = depth_conf.to(dtype=torch.float32).cpu()
                prediction["pose_enc"] = pose_enc.to(dtype=torch.float32).cpu()

                extri, intri = pose_encoding_to_extri_intri(prediction["pose_enc"], size_hw)
                prediction["extrinsic"] = extri.squeeze(0)   # CPU
                prediction["intrinsic"] = intri.squeeze(0)   # CPU

                prediction["world_points"] = unproject_depth_map_to_point_map(
                    prediction["depth"].squeeze(0),
                    extri.squeeze(0),
                    intri.squeeze(0),
                )

                if need_features:
                    prediction["features"] = feature_maps.to(dtype=torch.float32).cpu()

                # If you ever want to return patch_tokens, convert tensors to CPU here.
                # (Currently you don't store them in `prediction`, but this preserves your existing behavior.)
                for k, v in list(patch_tokens.items()):
                    if isinstance(v, torch.Tensor):
                        patch_tokens[k] = v.to(dtype=torch.float32).cpu()

                # cleanup references
                del aggregated_tokens_list, ps_idx, patch_tokens, pose_enc, depth_map, depth_conf, images
                if need_features:
                    del feature_maps

            else:  # Pi3
                prediction["extrinsic"] = remove_homogeneous_row(
                    torch.linalg.inv(res["camera_poses"].squeeze(0))
                ).to(dtype=torch.float32).cpu()

                prediction["world_points"] = res["points"].to(dtype=torch.float32).cpu().squeeze(0)

                depth_conf = res["conf"].to(dtype=torch.float32).cpu().squeeze(-1)
                prediction["depth_conf"] = torch.sigmoid(depth_conf)

                intrinsic, depth = estimate_intrinsics_and_depth(res["local_points"].squeeze(0))
                prediction["intrinsic"] = intrinsic.to(dtype=torch.float32).cpu().squeeze(0)

                prediction["depth"] = compute_depth(
                    prediction["world_points"],
                    prediction["extrinsic"],
                ).to(dtype=torch.float32).cpu().unsqueeze(0).unsqueeze(-1)

                del res, images

            # Convert all tensors to numpy arrays before putting in queue to avoid CUDA multiprocessing issues
            prediction_numpy = {}
            for key, value in prediction.items():
                if isinstance(value, torch.Tensor):
                    # Ensure tensor is detached and on CPU, then convert to numpy
                    prediction_numpy[key] = value.detach().cpu().numpy()
                    if prediction_numpy[key].shape[0] == 1:
                        prediction_numpy[key] = prediction_numpy[key].squeeze(0)
                else:
                    prediction_numpy[key] = value

            # Send result back with its global index so the main process can reorder.
            out_queue.put((idx, prediction_numpy))

            # Light cleanup (optional)
            gc.collect()

    # Optional: one cache clear at the very end
    torch.cuda.empty_cache()


# ------------------------------------------------------------
# Public API: multi-GPU inference with 4 GPUs.
# ------------------------------------------------------------
def run_inference_step_by_step_multi_gpu(
    make_model,  # Callable[[], torch.nn.Module] or ModelFactorySpec (picklable)
    batches: Sequence[torch.Tensor],
    size_hw: Tuple[int, int],
    need_features: bool = False,
    num_gpus: int = 2,
    amp_dtype: torch.dtype = torch.bfloat16,
) -> List[Dict[str, Any]]:
    """
    Parallelizes your existing per-item inference across `num_gpus` GPUs
    by spawning `num_gpus` processes (one per GPU) and sharding `batches`.

    Requirements:
      - `make_model()` must construct the model and load weights (in each process).
        (This is the most robust way; don't try to pass a live CUDA model across processes.)
      - `batches` should be a list/sequence of CPU tensors (optionally pin_memory()).

    Output matches your original function: a list of dict predictions, ordered by input.
    """
    assert num_gpus >= 1
    assert len(batches) >= 1

    # Optional: if batches are CPU tensors, pinning improves H2D throughput
    # (Safe even if already pinned)
    pinned_batches = []
    for x in batches:
        if isinstance(x, torch.Tensor) and x.device.type == "cpu":
            pinned_batches.append(x.pin_memory())
        else:
            pinned_batches.append(x)
    batches = pinned_batches

    n = len(batches)
    # Round-robin sharding keeps loads balanced if per-image cost varies.
    shards = [[] for _ in range(num_gpus)]
    for i in range(n):
        shards[i % num_gpus].append(i)

    ctx = mp.get_context("spawn")
    out_queue: mp.Queue = ctx.Queue()

    start = time.time()

    procs = []
    for rank in range(num_gpus):
        p = ctx.Process(
            target=_inference_worker,
            args=(
                rank,
                num_gpus,
                make_model,
                batches,
                shards[rank],
                size_hw,
                need_features,
                out_queue,
                amp_dtype,
            ),
        )
        p.daemon = False
        p.start()
        procs.append(p)

    # Collect results
    predictions: List[Optional[Dict[str, Any]]] = [None] * n
    received = 0
    while received < n:
        idx, pred = out_queue.get()
        # Convert numpy arrays back to torch tensors if needed (for consistency with original API)
        # But since downstream code expects numpy arrays (based on test_parallel.py line 132),
        # we keep them as numpy arrays
        predictions[idx] = pred
        received += 1

    # Join workers
    for p in procs:
        p.join()
        if p.exitcode != 0:
            raise RuntimeError(f"Worker exited with code {p.exitcode}")

    end = time.time()
    print(f"[INFERENCE][{num_gpus} GPUs] Time used: {end - start}s.")

    # Type narrowing
    return [p for p in predictions if p is not None]


# ------------------------------------------------------------
# Example: returns a picklable ModelFactorySpec for multiprocessing.
# ------------------------------------------------------------
def make_vggt_model_from_ckpt(
    ckpt_path: str,
    model_ctor: Callable[[], "VGGT"],
) -> ModelFactorySpec:
    """
    Returns a picklable ModelFactorySpec so multiprocessing can build the model
    in each worker. `model_ctor` is only used for type hint; we use "vggt" in the spec.
    """
    return ModelFactorySpec(ckpt_path, model_type="vggt")

def make_vggt():
    model = VGGT()

    return model



"""
USAGE (sketch):

# 1) Provide a top-level constructor for your model (args baked in).
def build_my_vggt() -> VGGT:
    model = VGGT(...)
    return model

make_model = make_vggt_model_from_ckpt("/path/to/ckpt.pt", build_my_vggt)

preds = run_inference_step_by_step_multi_gpu(
    make_model=make_model,
    batches=batches,          # list of CPU tensors
    size_hw=(H, W),
    need_features=False,
    num_gpus=4,
)
"""
