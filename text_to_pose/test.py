import os
import json
import numpy as np
import torch
import sys
sys.path.insert(0, '/home/nlp/rotemsh/transcription')

from text_to_pose.args import args
from text_to_pose.data import get_dataset
from text_to_pose.tokenizers.hamnosys.hamnosys_tokenizer import HamNoSysTokenizer
from text_to_pose.constants import DATASET_SIZE, num_steps_to_batch_size
from text_to_pose.model import IterativeTextGuidedPoseGenerationModel
from text_to_pose.pred import pred
from text_to_pose.metrics import get_poses_ranks
from pose_format import Pose
from pose_format.numpy import NumPyPoseBody
from shared.pose_utils import pose_hide_legs


def combine_results(experiment_name, results_path):
    results = dict()
    for file in os.listdir(results_path):
        if experiment_name in file:
            with open(os.path.join(results_path, file)) as f:
                results.update(json.load(f))
    return np.mean(list(results.values())), np.median(list(results.values()))


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    args.gpus = 1
    experiment_name = "reproduce_exclude_sep_2" #"exclude_sep_leave_out_pjm"
    #"exclude_bad_videos_sep_pos_embedding"
    args.num_steps = 10
    args.batch_size = num_steps_to_batch_size[args.num_steps]
    print("experiment_name:", experiment_name)

    test_size = int(0.1*DATASET_SIZE)
    dataset = get_dataset(name=args.dataset, poses=args.pose, fps=args.fps,
                                       components=args.pose_components, exclude=True,
                                       max_seq_size=args.max_seq_size,
                                       split=f'test[:{test_size}]')
                                       # leave_out="pjm")

    _, num_pose_joints, num_pose_dims = dataset[0]["pose"]["data"].shape
    pose_header = dataset.data[0]["pose"].header

    # Model Arguments
    model_args = dict(tokenizer=HamNoSysTokenizer(),
                      pose_dims=(num_pose_joints, num_pose_dims),
                      hidden_dim=128,#256,#args.hidden_dim,
                      text_encoder_depth=args.text_encoder_depth,
                      pose_encoder_depth=args.pose_encoder_depth,
                      encoder_heads=args.encoder_heads, #4,
                      # encoder_dim_feedforward=512,
                      max_seq_size=args.max_seq_size,
                      num_steps=args.num_steps,
                      tf_p=args.tf_p,
                      masked_loss=args.masked_loss,
                      separate_positional_embedding=args.separate_positional_embedding,
                      num_pose_projection_layers=args.num_pose_projection_layers,
                      do_pose_self_attention=False,#True,
                      use_transformer_decoder=False,#True,
                      concat=True)

    args.checkpoint = f"/home/nlp/rotemsh/transcription/models/{experiment_name}/model.ckpt"
    model = IterativeTextGuidedPoseGenerationModel.load_from_checkpoint(args.checkpoint, **model_args)
    model.eval()

    output_dir = f"/home/nlp/rotemsh/transcription/text_to_pose/videos/{experiment_name}"

    visualize = False
    if visualize:
        pred(model, dataset, os.path.join(output_dir, "test_results"), vis_pred_only=False, gen_k=200)

    test_dtw = True
    if test_dtw:
        keypoints_path = "/home/nlp/rotemsh/SLP/data/keypoints_dir"
        keypoints_dirs = os.listdir(keypoints_path)
        with open("/home/nlp/rotemsh/transcription/datasets/hamnosys/data.json", 'r') as f:
            data = json.load(f)
            data_ids = list(filter(lambda x: x in keypoints_dirs, data.keys()))

        model = model.cuda()
        with torch.no_grad():
            ds = dataset
            # for ds_name, ds in {"lsf": dataset_lsf, "rest": dataset}.items():
            rank_1_pred_sum = rank_5_pred_sum = rank_10_pred_sum = rank_1_label_sum = rank_5_label_sum =  \
                rank_10_label_sum = 0
            pred2label_distances = dict()
            for datum in ds:
                first_pose = datum["pose"]["data"][0]
                seq_len = int(datum["pose"]["length"].item()) #if args.num_steps != 10 else -1
                seq_iter = model.forward(text=datum["text"], first_pose=first_pose.cuda(), sequence_length=seq_len)
                for j in range(model.num_steps):
                    seq = next(seq_iter)

                data = torch.unsqueeze(seq, 1).cpu()
                conf = torch.ones_like(data[:, :, :, 0])
                pose_body = NumPyPoseBody(25, data.numpy(), conf.numpy())
                predicted_pose = Pose(pose_header, pose_body)
                pose_hide_legs(predicted_pose)
                pred2label_distance, rank_1_pred, rank_5_pred, rank_10_pred, rank_1_label, rank_5_label, \
                rank_10_label = get_poses_ranks(predicted_pose, datum["id"], keypoints_path, data_ids)
                pred2label_distances[datum["id"]] = pred2label_distance
                rank_1_pred_sum += int(rank_1_pred)
                rank_5_pred_sum += int(rank_5_pred)
                rank_10_pred_sum += int(rank_10_pred)
                rank_1_label_sum += int(rank_1_label)
                rank_5_label_sum += int(rank_5_label)
                rank_10_label_sum += int(rank_10_label)

            num_samples = len(ds)
            print(f"rank 1 pred sum: {rank_1_pred_sum} / {num_samples}: {rank_1_pred_sum / num_samples}")
            print(f"rank 5 pred sum: {rank_5_pred_sum} / {num_samples}: {rank_5_pred_sum / num_samples}")
            print(f"rank 10 pred sum: {rank_10_pred_sum} / {num_samples}: {rank_10_pred_sum / num_samples}")

            print(f"rank 1 label sum: {rank_1_label_sum} / {num_samples}: {rank_1_label_sum / num_samples}")
            print(f"rank 5 label sum: {rank_5_label_sum} / {num_samples}: {rank_5_label_sum / num_samples}")
            print(f"rank 10 label sum: {rank_10_label_sum} / {num_samples}: {rank_10_label_sum / num_samples}")

            with open(f"results/pred2label_distances_gt_seq_len_all.json",#{ds_name}.json",
                      'w') as f:
                json.dump(pred2label_distances, f)

            print(f"mean distance between pred and label: {np.mean(list(pred2label_distances.values()))}")
            print(f"median distance between pred and label: {np.median(list(pred2label_distances.values()))}")

    do_combine_results = False
    if do_combine_results:
        base_results_path = "/home/nlp/rotemsh/transcription/results"
        leave_out_language = "pjm"
        print(combine_results(f"gt_seq_len_no_{leave_out_language}_rest", base_results_path))
        print(combine_results(f"gt_seq_len_no_{leave_out_language}_{leave_out_language}", base_results_path))
