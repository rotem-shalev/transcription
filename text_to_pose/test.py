import os
import torch
import json
import sys
sys.path.insert(0, '/home/nlp/rotemsh/transcription')

# from shared.collator import zero_pad_collator
from text_to_pose.args import args
from text_to_pose.data import get_dataset
from text_to_pose.tokenizers.hamnosys.hamnosys_tokenizer import HamNoSysTokenizer
from text_to_pose.constants import DATASET_SIZE
from text_to_pose.model import IterativeTextGuidedPoseGenerationModel
from text_to_pose.pred import pred
from text_to_pose.metrics import compare_poses
from pose_format import Pose
from pose_format.numpy import NumPyPoseBody
from shared.pose_utils import pose_normalization_info, pose_hide_legs

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"#"0"
    args.gpus = 1
    experiment_name = "reproduce_exclude_sep_2" #"exclude_bad_videos_sep_pos_embedding"
    print("experiment_name:", experiment_name)

    test_size = int(0.1*DATASET_SIZE)
    # train_split = f'test[{test_size}:]+train'
    # test_split = f'test[:{test_size}]'
    # train_dataset = get_dataset(name=args.dataset, poses=args.pose, fps=args.fps,
    #                             components=args.pose_components, exclude=True,
    #                             max_seq_size=args.max_seq_size, split=train_split)
    # validation_dataset = get_dataset(name=args.dataset, poses=args.pose, fps=args.fps,
    #                                  components=args.pose_components, exclude=True,
    #                                  max_seq_size=args.max_seq_size, split=test_split)
    #
    dataset = get_dataset(name=args.dataset, poses=args.pose, fps=args.fps,
                                components=args.pose_components, exclude=True,
                                max_seq_size=args.max_seq_size, split=f'test[7:20]')#{test_size}]')#+train")

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

    output_dir = f"/home/nlp/rotemsh/transcription/text_to_pose/videos/{experiment_name}_2"

    visualize = False
    if visualize:
        pred(model, dataset, os.path.join(output_dir, "metric_test_pred_only"), vis_pred_only=True, gen_k=200)

    test_dtw = True
    if test_dtw:
        keypoints_path = "/home/nlp/rotemsh/SLP/data/keypoints_dir"

        with open("/home/nlp/rotemsh/transcription/datasets/hamnosys/data.json", 'r') as f:
            data = json.load(f)
            data_ids = data.keys()  # TODO- use all or only test?

        model.eval()
        with torch.no_grad():
            rank_1_pred_sum = rank_5_pred_sum = rank_10_pred_sum = rank_1_label_sum = rank_5_label_sum = \
                rank_10_label_sum = 0
            for i, datum in enumerate(dataset):
                first_pose = datum["pose"]["data"][0]
                seq_len = int(datum["pose"]["length"].item())  # TODO- change to trained model pred
                seq_iter = model.forward(text=datum["text"], first_pose=first_pose, sequence_length=seq_len)
                for j in range(model.num_steps):
                    seq = next(seq_iter)

                data = torch.unsqueeze(seq, 1).cpu()
                conf = torch.ones_like(data[:, :, :, 0])
                pose_body = NumPyPoseBody(25, data.numpy(), conf.numpy())
                predicted_pose = Pose(pose_header, pose_body)
                pose_hide_legs(predicted_pose)
                # normalization_info = pose_normalization_info(predicted_pose.header)
                # predicted_pose = predicted_pose.normalize(normalization_info, scale_factor=100)
                # predicted_pose.focus()

                rank_1_pred, rank_5_pred, rank_10_pred, rank_1_label, rank_5_label, rank_10_label = compare_poses(
                    predicted_pose, datum["id"], keypoints_path, data_ids)
                rank_1_pred_sum += int(rank_1_pred)
                rank_5_pred_sum += int(rank_5_pred)
                rank_10_pred_sum += int(rank_10_pred)
                rank_1_label_sum += int(rank_1_label)
                rank_5_label_sum += int(rank_5_label)
                rank_10_label_sum += int(rank_10_label)

            num_samples = len(dataset)
            print(f"rank 1 pred sum: {rank_1_pred_sum} / {num_samples}: {rank_1_pred_sum / num_samples}")
            print(f"rank 5 pred sum: {rank_5_pred_sum} / {num_samples}: {rank_5_pred_sum / num_samples}")
            print(f"rank 10 pred sum: {rank_10_pred_sum} / {num_samples}: {rank_10_pred_sum / num_samples}")

            print(f"rank 1 label sum: {rank_1_label_sum} / {num_samples}: {rank_1_label_sum / num_samples}")
            print(f"rank 5 label sum: {rank_5_label_sum} / {num_samples}: {rank_5_label_sum / num_samples}")
            print(f"rank 10 label sum: {rank_10_label_sum} / {num_samples}: {rank_10_label_sum / num_samples}")
