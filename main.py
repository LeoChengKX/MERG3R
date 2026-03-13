import torch
import numpy as np
import os
import argparse
import json

from algos.utils import *
from algos.sequence import create_sequence
from algos.bundle_adjustment import global_bundle_adjustment
from algos.alignment import align_extrinsics
from algos.tracking import  extract_matches_lightglue, graph_extract_matches_lightglue


import gc

def parse_args():
    parser = argparse.ArgumentParser("Test parallel inference on VGGT. ")
    parser.add_argument("--subset_size", type=int, default=55)
    parser.add_argument("--num_images", type=int, default=-1)
    parser.add_argument("--overlap", type=int, default=5)
    parser.add_argument("--subsample", type=int, default=1)
    parser.add_argument("--write_local", action="store_true")
    parser.add_argument("--ba_type", type=str, default="gradient")
    parser.add_argument("--alignment_type", type=str, default="weighted_iterative")
    parser.add_argument("--global_ba", action="store_true")
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--sequence_type", type=str, default="video")
    parser.add_argument("--lr", type=float, default=1e-4)  # 3e-4
    parser.add_argument("--epoch", type=int, default=300)
    parser.add_argument("--max_reproj", type=float, default=8.0)
    parser.add_argument("--stride", type=int, default=100)
    parser.add_argument("--model", type=str, default="vggt")
    parser.add_argument("--multi_dirs", action="store_true")
    parser.add_argument("--point_vis_threshold", type=float, default=50.0)
    parser.add_argument("--tracking_type", type=str, default="graph", choices=['graph', 'video'])
    parser.add_argument("--alpha", type=float, default=0.8)
    parser.add_argument("--splitting_type", type=str, default="interleave", choices=['interleave', 'zigzag', 'threshold', "original", "original_threshold"])
    args = parser.parse_args()

    
    return args


def main():
    
    args = parse_args()

    args_dict = vars(args)
    
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run this pipeline. No CUDA device was detected.")
    device = "cuda"

    print(f"[MAIN] Output to {args.output_dir}")
    subsample = args.subsample

    os.makedirs(args.output_dir, exist_ok=True)
    # Save to JSON
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(args_dict, f, indent=2)

    images, image_names = process_images(args.dataset, subsample, device, args.num_images, args.multi_dirs)

    size_hw = images.shape[-2:]

    
    start_time = time.time()

    # Reset and start GPU memory tracking
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    print("[MAIN] Create sequence. ")

    seq_start = time.time()
    sequence = create_sequence(images, image_names, 
                                sequence_type=args.sequence_type, 
                                subset_size=args.subset_size, 
                                overlap=args.overlap, 
                                save_path=args.output_dir, 
                                alpha=args.alpha, 
                                splitting_type=args.splitting_type)
    seq_end = time.time()

    batches = sequence.image_split

    for i, img in enumerate(batches):
        batches[i] = img.to("cpu")

    print(f"[MAIN] Number of subsets {len(batches)}. ")
    gc.collect()
    torch.cuda.empty_cache()

    
    model, _ = load_model(args.model, device=device)
    inf_start = time.time()

    sequence.predictions = run_inference_step_by_step(model, batches, size_hw, device, need_features=False)
    inf_end = time.time()
    del model


    gc.collect()
    torch.cuda.empty_cache()
    # Convert each prediction to extrinsics
    for i, prediction in enumerate(sequence.predictions):
        prediction['images'] = batches[i]

    # Squeeze and convert to numpy
    for prediction in sequence.predictions:
        for key in prediction.keys():
            if isinstance(prediction[key], torch.Tensor):
                prediction[key] = prediction[key].cpu().numpy() 
                if prediction[key].shape[0] == 1:
                    prediction[key] = prediction[key].squeeze(0) # remove batch dimension
    
    
    final_predictions, _, _ = align_extrinsics(sequence, method=args.alignment_type, ba=False)
   
    restore_predictions_order(final_predictions)
    for key in sequence.subset_to_img_ids.keys():
        if isinstance(sequence.subset_to_img_ids[key], torch.Tensor):
            sequence.subset_to_img_ids[key] = sequence.subset_to_img_ids[key].cpu().numpy()
    print("[MAIN] DONE MERGING")

    tracking_start = 0
    tracking_end = 0

    ba_start = 0
    ba_end = 0
    if args.global_ba:

        print("START TRACKING")

        final_predictions['world_points'] = unproject_depth_map_to_point_map(final_predictions['depth'], final_predictions['extrinsic'], final_predictions['intrinsic'])

        tracking_start = time.time()

        if args.sequence_type == 'video':
            track, points_id, points_3d, points_conf = extract_matches_lightglue(images, final_predictions['world_points'], final_predictions['depth_conf'],
                                                                                    final_predictions['extrinsic'], final_predictions['intrinsic'],
                                                                                    steps=[1, 2, 3, 5, 7, 10], max_num_keypoints=4096, device=device)
        elif args.sequence_type == 'shortest_path':

            if args.tracking_type =="graph":
                track, points_id, points_3d, points_conf = graph_extract_matches_lightglue(images, final_predictions['world_points'], 
                                                                        final_predictions['depth_conf'], 
                                                                        final_predictions['extrinsic'], final_predictions['intrinsic'], k=5,
                                                                        max_num_keypoints=4096, device=device)
            elif args.tracking_type == 'video':
                reordered_images = images[sequence.video_path]
                reordered_points = final_predictions['world_points'][sequence.video_path]
                reordered_conf = final_predictions['depth_conf'][sequence.video_path]
                reordered_extrinsic = final_predictions['extrinsic'][sequence.video_path]
                reordered_intrinsic = final_predictions['intrinsic'][sequence.video_path]

                track, points_id, points_3d, points_conf = extract_matches_lightglue(reordered_images, reordered_points, reordered_conf, 
                                                                                    reordered_extrinsic, reordered_intrinsic,
                                                                                    steps=[1, 2, 3, 5, 7, 10], max_num_keypoints=4096, device=device)
            
                inverse_path = np.argsort(sequence.video_path).tolist()

                # Reorder both lists
                track = [track[i] for i in inverse_path]
                points_id = [points_id[i] for i in inverse_path]

        tracking_end = time.time()
        del final_predictions['world_points']
        final_predictions['track'] = track
        final_predictions['points_id'] = points_id
        final_predictions['points'] = points_3d


        print("START GLOBAL BA")
        
        ba_start = time.time()
        _, _ =  global_bundle_adjustment(final_predictions, 
                                        track, 
                                        points_id, 
                                        points_3d, 
                                        points_conf, 
                                        max_reproj_error=args.max_reproj, 
                                        lr=args.lr,
                                        epoch=args.epoch
                                        )

        ba_end = time.time()

    
    torch.cuda.synchronize()
    end_time = time.time()

    peak_mem = torch.cuda.max_memory_allocated() / (1024**2)  # in MiB
    elapsed = end_time - start_time

    final_predictions['world_points'] = unproject_depth_map_to_point_map(final_predictions['depth'], final_predictions['extrinsic'], final_predictions['intrinsic'])

    with open(os.path.join(args.output_dir, "computation_stats.txt",), "w")as f:
        f.write(f"Runtime: {elapsed:.4f} seconds\n")
        f.write(f"Sequence Time: {seq_end - seq_start} seconds\n")
        f.write(f"Inference Time: {inf_end - inf_start} seconds\n")
        f.write(f"Tracking Time: {tracking_end - tracking_start} seconds\n")
        f.write(f"BA Time: {ba_end - ba_start} seconds\n")
        f.write(f"Peak GPU memory: {peak_mem:.2f} MiB\n")

    write_recon_to_colmap(args.output_dir, final_predictions, sequence.images, sequence.image_names, stride=args.stride, conf_threshold=args.point_vis_threshold)
    

if __name__ == "__main__":
    main()
