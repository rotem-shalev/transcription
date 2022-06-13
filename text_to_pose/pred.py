import os
import shutil
from typing import List
import cv2
from tqdm import tqdm
import numpy as np
import torch
import matplotlib.pyplot as plt

from pose_format import Pose
from pose_format.numpy import NumPyPoseBody
from pose_format.pose_visualizer import PoseVisualizer

from shared.pose_utils import pose_normalization_info, pose_hide_legs
from text_to_pose.args import args
from text_to_pose.data import get_dataset
from text_to_pose.model import IterativeTextGuidedPoseGenerationModel
from text_to_pose.tokenizers.hamnosys.hamnosys_tokenizer import HamNoSysTokenizer

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Only use CPU


def visualize_pose(poses: List[Pose], pose_name: str, output_dir: str):
    f_name = os.path.join(output_dir, pose_name)
    width = max(poses[0].header.dimensions.width, poses[1].header.dimensions.width) * 2
    height = max(poses[0].header.dimensions.height, poses[1].header.dimensions.height) + 40

    font = cv2.FONT_HERSHEY_SIMPLEX
    color = (0, 250, 0)
    frames = None

    for pose in poses:
        normalization_info = pose_normalization_info(pose.header)

        # Normalize pose
        pose = pose.normalize(normalization_info, scale_factor=100)
        pose.focus()
        # width and height may change after normalization
        width = max(width, pose.header.dimensions.width*2)
        height = max(height, pose.header.dimensions.height+40)

        # Draw pose
        visualizer = PoseVisualizer(pose, thickness=2)
        if frames is None:
            frames = list(visualizer.draw())
        else:
            image_size = (width, height)
            out = cv2.VideoWriter(f_name, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), poses[0].body.fps, image_size)
            for i, frame in enumerate(tqdm(visualizer.draw())):
                empty = np.full((image_size[1], image_size[0]//2, 3), 255, dtype=np.uint8)
                if len(frames) > i:
                    empty[40:frames[i].shape[0]+40, :frames[i].shape[1]] = frames[i]
                label_frame = empty
                empty2 = np.full((image_size[1], image_size[0]//2, 3), 255, dtype=np.uint8)
                empty2[40:frame.shape[0]+40, :frame.shape[1]] = frame
                frame = empty2

                label_pred_im = np.concatenate((label_frame, frame), axis=1)

                label_pred_im = cv2.putText(label_pred_im, "label", (image_size[0]//5, 30), font, 1, color, 2, 0)
                label_pred_im = cv2.putText(label_pred_im, "pred", (image_size[0]//5 + image_size[0]//2, 30),
                                            font, 1, color, 2, 0)
                # plt.imshow(label_pred_im)
                # plt.show()
                out.write(label_pred_im)

            out.release()
    # visualizer.save_video(os.path.join(args.pred_output, pose_name),
    #                       visualizer.draw(),
    #                       custom_ffmpeg=None)


def visualize_poses(_id: str, text: str, poses: List[Pose]) -> str:
    lengths = " / ".join([str(len(p.body.data)) for p in poses])
    html_tags = f"<h3><u>{_id}</u>: <span class='hamnosys'>{text}</span> ({lengths})</h3>"

    # for k, pose in enumerate(poses):
    pose_name = f"{_id}.mp4"
    visualize_pose(poses, pose_name, args.pred_output)
    html_tags += f"<video src='{pose_name}' controls preload='none'></video>"

    return html_tags


def pred(model, dataset, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    _, num_pose_joints, num_pose_dims = dataset[0]["pose"]["data"].shape
    pose_header = dataset.data[0]["pose"].header

    model.eval()
    with torch.no_grad():
        for datum in dataset:
            first_pose = datum["pose"]["data"][0]
            seq_iter = model.forward(text=datum["text"], first_pose=first_pose)
            for i in range(model.num_steps):
                seq = next(seq_iter)

            data = torch.unsqueeze(seq, 1).cpu()
            conf = torch.ones_like(data[:, :, :, 0])
            pose_body = NumPyPoseBody(args.fps, data.numpy(), conf.numpy())
            predicted_pose = Pose(pose_header, pose_body)
            pose_hide_legs(predicted_pose)

            pose_name = f"{datum['id']}.mp4"
            visualize_pose([datum["pose"]["obj"], predicted_pose], pose_name, output_dir)


if __name__ == '__main__':
    args.checkpoint = "/home/nlp/rotemsh/transcription/text_to_pose/models/20_steps/model-v2.ckpt"
    args.pred_output = "/home/nlp/rotemsh/transcription/videos/20_steps"
    args.ffmpeg_path = "/home/nlp/rotemsh/ffmpeg-5.0.1-amd64-static/"

    if args.checkpoint is None:
        raise Exception("Must specify `checkpoint`")
    if args.pred_output is None:
        raise Exception("Must specify `pred_output`")
    if args.ffmpeg_path is None:
        raise Exception("Must specify `ffmpeg_path`")

    os.makedirs(args.pred_output, exist_ok=True)

    dataset = get_dataset(name=args.dataset, poses=args.pose, fps=args.fps, components=args.pose_components,
                          max_seq_size=args.max_seq_size, split="train[:20]")

    _, num_pose_joints, num_pose_dims = dataset[0]["pose"]["data"].shape
    pose_header = dataset.data[0]["pose"].header

    model_args = dict(tokenizer=HamNoSysTokenizer(),
                      pose_dims=(num_pose_joints, num_pose_dims),
                      hidden_dim=args.hidden_dim,
                      text_encoder_depth=args.text_encoder_depth,
                      pose_encoder_depth=args.pose_encoder_depth,
                      encoder_heads=args.encoder_heads,
                      max_seq_size=args.max_seq_size,
                      num_steps=args.num_steps)

    model = IterativeTextGuidedPoseGenerationModel.load_from_checkpoint(args.checkpoint, **model_args)
    model.eval()

    html = []

    with torch.no_grad():
        for datum in dataset:
            first_pose = datum["pose"]["data"][0]
            # datum["text"] = ""
            seq_iter = model.forward(text=datum["text"], first_pose=first_pose)
            for i in range(10):  # This loop is near instantaneous
                seq = next(seq_iter)

            data = torch.unsqueeze(seq, 1).cpu()
            conf = torch.ones_like(data[:, :, :, 0])
            pose_body = NumPyPoseBody(args.fps, data.numpy(), conf.numpy())
            predicted_pose = Pose(pose_header, pose_body)
            pose_hide_legs(predicted_pose)

            html.append(visualize_poses(_id=datum["id"],
                                        text=datum["text"],
                                        poses=[datum["pose"]["obj"], predicted_pose]))

        # # Iterative change
        # datum = dataset[12]  # dataset[0] starts with an empty frame
        # first_pose = datum["pose"]["data"][0]
        # seq_iter = model.forward(text=datum["text"], first_pose=first_pose, step_size=1)
        #
        # data = torch.stack([next(seq_iter) for i in range(1000)], dim=1)
        # data = data[:, ::100, :, :]
        #
        # conf = torch.ones_like(data[:, :, :, 0])
        # pose_body = NumPyPoseBody(args.fps, data.numpy(), conf.numpy())
        # predicted_pose = Pose(pose_header, pose_body)
        # pose_hide_legs(predicted_pose)
        # predicted_pose.focus()
        # # shift poses
        # for i in range(predicted_pose.body.data.shape[1] - 1):
        #     max_x = np.max(predicted_pose.body.data[:, i, :, 0])
        #     predicted_pose.body.data[:, i + 1, :, 0] += max_x
        #
        # html.append(visualize_poses(_id=datum["id"] + "_iterative",
        #                             text=datum["text"],
        #                             poses=[datum["pose"]["obj"], predicted_pose]))

    with open(os.path.join(args.pred_output, "index.html"), "w", encoding="utf-8") as f:
        f.write(
            "<style>@font-face {font-family: HamNoSys;src: url(HamNoSys.ttf);}.hamnosys{font-family: HamNoSys}</style>")
        f.write("<br><br><br>".join(html))

    shutil.copyfile(model.tokenizer.font_path, os.path.join(args.pred_output, "HamNoSys.ttf"))
