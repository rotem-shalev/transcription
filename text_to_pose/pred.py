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
from text_to_pose.tokenizers.hamnosys.hamnosys_tokenizer import HamNoSysTokenizer

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Only use CPU


def visualize_pose(poses: List[Pose], pose_name: str, output_dir: str, slow: bool = False):
    os.makedirs(output_dir, exist_ok=True)
    f_name = os.path.join(output_dir, pose_name)
    fps = poses[0].body.fps
    if slow:
        f_name = f_name[:-4] + "_slow" + ".mp4"
        fps = poses[0].body.fps // 2

    frames = []
    for i in range(len(poses)):
        # Normalize pose
        normalization_info = pose_normalization_info(poses[i].header)
        pose = poses[i].normalize(normalization_info, scale_factor=100)
        pose.focus()

        visualizer = PoseVisualizer(pose, thickness=2)
        frames.append(list(visualizer.draw()))

    if len(poses) == 1:
        image_size = (frames[0][0].shape[1], frames[0][0].shape[0])
        out = cv2.VideoWriter(f_name,
                              cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps,
                              image_size)
        for frame in frames[0]:
            out.write(frame)
        out.release()
        return

    image_size = (max(frames[0][0].shape[1], frames[1][0].shape[1]) * 2,
                  max(frames[0][0].shape[0], frames[1][0].shape[0]) + 40)
    out = cv2.VideoWriter(f_name,
                          cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps,
                          image_size)
    for i, frame in enumerate(frames[1]):
        empty = np.full((image_size[1], image_size[0]//2, 3), 255, dtype=np.uint8)
        if len(frames[0]) > i:
            empty[40:frames[0][i].shape[0]+40, :frames[0][i].shape[1]] = frames[0][i]
        label_frame = empty
        empty2 = np.full((image_size[1], image_size[0]//2, 3), 255, dtype=np.uint8)
        empty2[40:frame.shape[0]+40, :frame.shape[1]] = frame
        pred_frame = empty2
        label_pred_im = concat_and_add_label(label_frame, pred_frame, image_size)
        out.write(label_pred_im)

    if i < len(frames[0]):
        for frame in frames[0][i:]:
            empty = np.full((image_size[1], image_size[0] // 2, 3), 255, dtype=np.uint8)
            empty[40:frame.shape[0] + 40, :frame.shape[1]] = frame
            label_frame = empty
            pred_frame = np.full((image_size[1], image_size[0] // 2, 3), 255, dtype=np.uint8)
            label_pred_im = concat_and_add_label(label_frame, pred_frame, image_size)
            out.write(label_pred_im)

    out.release()


def concat_and_add_label(label_frame, pred_frame, image_size):
    font = cv2.FONT_HERSHEY_SIMPLEX
    color = (0, 250, 0)
    label_pred_im = np.concatenate((label_frame, pred_frame), axis=1)
    label_pred_im = cv2.putText(label_pred_im, "label", (image_size[0] // 5, 30), font, 1,
                                color, 2, 0)
    label_pred_im = cv2.putText(label_pred_im, "pred",
                                (image_size[0] // 5 + image_size[0] // 2, 30),
                                font, 1, color, 2, 0)
    return label_pred_im


def visualize_poses(_id: str, text: str, poses: List[Pose]) -> str:
    lengths = " / ".join([str(len(p.body.data)) for p in poses])
    html_tags = f"<h3><u>{_id}</u>: <span class='hamnosys'>{text}</span> ({lengths})</h3>"

    pose_name = f"{_id}.mp4"
    visualize_pose(poses, pose_name, args.pred_output)
    html_tags += f"<video src='{pose_name}' controls preload='none'></video>"

    return html_tags


def visualize_seq(seq, pose_header, output_dir, id, label_pose=None, fps=25):
    with torch.no_grad():
        data = torch.unsqueeze(seq, 1).cpu()
        conf = torch.ones_like(data[:, :, :, 0])
        pose_body = NumPyPoseBody(fps, data.numpy(), conf.numpy())
        predicted_pose = Pose(pose_header, pose_body)
        pose_hide_legs(predicted_pose)

        pose_name = f"{id}.mp4"
        visualize_pose([label_pose, predicted_pose], pose_name, output_dir)


def vis_label_only(dataset, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    _, num_pose_joints, num_pose_dims = dataset[0]["pose"]["data"].shape
    for i, datum in enumerate(dataset):
        pose_name = f"{datum['id']}_label_only.mp4"
        label_pose = datum["pose"]["obj"]
        visualize_pose([label_pose], pose_name, output_dir)


def pred(model, dataset, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    _, num_pose_joints, num_pose_dims = dataset[0]["pose"]["data"].shape
    pose_header = dataset.data[0]["pose"].header

    model.eval()
    with torch.no_grad():
        for i, datum in enumerate(dataset):
            if ("train" in output_dir and i > 20) or i > 30:
                break
            first_pose = datum["pose"]["data"][0]
            seq_iter = model.forward(text=datum["text"], first_pose=first_pose)
            for i in range(model.num_steps):
                seq = next(seq_iter)
            visualize_seq(seq, pose_header, output_dir, datum["id"], datum["pose"]["obj"])


def add_orig_vid_to_model_vid(orig_vids_path, model_vids_path):
    os.makedirs(os.path.join(model_vids_path, "combined"), exist_ok=True)
    for vid in os.listdir(model_vids_path):
        if vid == "combined":
            continue
        cap_orig = cv2.VideoCapture(os.path.join(orig_vids_path, vid))
        cap_keypoints = cv2.VideoCapture(os.path.join(model_vids_path, vid))
        _, frame_orig = cap_orig.read()
        _, frame_keypoints = cap_keypoints.read()
        if frame_orig is None or frame_keypoints is None:
            print(vid)
            continue
        scale = False
        if frame_orig.shape[0] > 1.5 * frame_keypoints.shape[0]:
            scale_factor = frame_orig.shape[0] // frame_keypoints.shape[0]
            frame_orig = cv2.resize(frame_orig, (frame_orig.shape[1] // scale_factor,
                                                 frame_orig.shape[0] // scale_factor))
            scale = True
        height = max(frame_orig.shape[0], frame_keypoints.shape[0])
        out_vid = cv2.VideoWriter(os.path.join(model_vids_path, "combined", vid),
                                  cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 25,
                                  (frame_orig.shape[1] + frame_keypoints.shape[1], height))
        i = 1
        while cap_orig.isOpened():
            if height == frame_orig.shape[0]:
                empty = np.full((height, frame_keypoints.shape[1], 3), 255, dtype=np.uint8)
                empty[-frame_keypoints.shape[0]:, :] = frame_keypoints
                cat = np.concatenate((frame_orig, empty), axis=1)
            else:
                empty = np.full((height, frame_orig.shape[1], 3), 255, dtype=np.uint8)
                empty[-frame_orig.shape[0]:, :] = frame_orig
                cat = np.concatenate((empty, frame_keypoints), axis=1)
            out_vid.write(cat)
            ret, frame_orig = cap_orig.read()
            ret1, frame_keypoints = cap_keypoints.read()
            if not ret or not ret1:
                break
            if scale:
                frame_orig = cv2.resize(frame_orig, (frame_orig.shape[1] // scale_factor,
                                                     frame_orig.shape[0] // scale_factor))
            i += 1
        out_vid.release()


if __name__ == '__main__':
    from text_to_pose.model import IterativeTextGuidedPoseGenerationModel

    # orig_vids_path = "/home/nlp/rotemsh/transcription/datasets/videos"
    # model_vids_path = "/home/nlp/rotemsh/transcription/text_to_pose/videos/learned_first_pose_tf_step_level/val_+500"
    # add_orig_vid_to_model_vid(orig_vids_path, model_vids_path)
    import json
    with open("/home/nlp/rotemsh/transcription/shared/pjm_left_videos.json", 'r') as f:
        PJM_LEFT_VIDEOS_LST = json.load(f)

    dataset = get_dataset(name=args.dataset, poses=args.pose, fps=args.fps,
                          components=args.pose_components,
                          max_seq_size=args.max_seq_size, split="train[1000:]", no_flip=False)
    pjm_data = [d for d in dataset if d["id"] in ["pjm_3191", "pjm_1717", "pjm_193", "pjm_1730", "pjm_1953",
                                                  "pjm_2558", "pjm_2740"]]
    for datum in dataset:
        if datum["id"] in ["pjm_3191", "pjm_1717", "pjm_193", "pjm_1730", "pjm_1953", "pjm_2558", "pjm_2740"]:
            print(datum["id"])
            pose_name = f"{datum['id']}_left.mp4"
            label_pose = datum["pose"]["obj"]
            visualize_pose([label_pose], pose_name, "videos/lefties")
    exit()

    from shared.collator import zero_pad_collator
    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset, batch_size=args.batch_size,  # num_workers=8,
               shuffle=True, collate_fn=zero_pad_collator)
    pose_header = dataset.data[0]["pose"].header

    _, num_pose_joints, num_pose_dims = dataset[0]["pose"]["data"].shape
    args.num_steps = 10
    num_steps_to_batch_size = {10: 16, 50: 8, 100: 4}
    batch_size_to_accumulate = {16: 2, 8: 4, 4: 8}
    args.batch_size = num_steps_to_batch_size[args.num_steps]
    args.tf_p = 0.5
    model_args = dict(tokenizer=HamNoSysTokenizer(),
                      pose_dims=(num_pose_joints, num_pose_dims),
                      hidden_dim=args.hidden_dim,
                      text_encoder_depth=args.text_encoder_depth,
                      pose_encoder_depth=args.pose_encoder_depth,
                      encoder_heads=args.encoder_heads,
                      max_seq_size=args.max_seq_size,
                      num_steps=args.num_steps,
                      tf_p=args.tf_p)
    experiment_name = "learned_first_pose_tf_step_level"
    args.checkpoint = f"/home/nlp/rotemsh/transcription/models/{experiment_name}/model-v1.ckpt"
    model = IterativeTextGuidedPoseGenerationModel.load_from_checkpoint(args.checkpoint, **model_args)
    model.eval()

    type2ham = {"apple": "\ue005\ue011\ue00c\ue027\ue03e\ue04e\ue0d1\ue0e2\ue089\ue0ba\ue0c6\ue0aa\ue028\ue038\ue0e3",
                "appreciate": "\ue0e9\ue000\ue00c\ue031\ue03e\ue0d1\ue0a6",
                "area": "\ue005\ue00c\ue029\ue0e6\ue028\ue03c\ue095\ue0c6\ue0d8",
                "army": "\ue001\ue00d\ue020\ue0e6\ue027\ue03c\ue059\ue051\ue0d1\ue082\ue0bb\ue0ca\ue051\ue059\ue0d1",
                "arrival": "\ue0e8\ue0e2\ue001\ue00c\ue010\ue0e7\ue001\ue00c\ue0e3\ue0e2\ue020\ue03d\ue0e7\ue02a\ue038"
                          "\ue0e3\ue0e2\ue040\ue058\ue0e7\ue052\ue0e3\ue0e2\ue0e2\ue084\ue0ba\ue0e7\ue0af\ue0e3\ue0aa\ue0e2\ue028\ue03c\ue0e7\ue0af\ue0e3\ue0e2\ue067\ue0e7\ue077\ue072\ue0e3\ue0d1\ue0e3",
                "ash": "\ue0e9\ue007\ue028\ue03d\ue0e2\ue071\ue0e7\ue071\ue0e3\ue0d1\ue051\ue0e2\ue082\ue0a4\ue0e3",
                "ask":  "\ue008\ue020\ue03e\ue04a\ue0d1\ue089\ue0ba",
                "ultimately": "\ue0e9\ue000\ue00c\ue029\ue03f\ue051\ue0aa\ue03c\ue052",
                "aunt": "\ue003\ue00d\ue011\ue020\ue0e6\ue027\ue03e\ue04d\ue0d0\ue08c\ue0c6\ue0d8",
                "authority": "\ue0e9\ue002\ue00d\ue026\ue03e\ue0a6\ue0c7\ue0d8",
                "fall": "\ue0e8\ue001\ue0e6\ue005\ue00c\ue020\ue03c\ue050\ue0e2\ue084\ue0bd\ue0c9\ue0a4\ue0aa\ue031"
                        "\ue0e3",
                "back": "\ue002\ue00d\ue026\ue03c\ue050\ue0e6\ue04f\ue058\ue0d1\ue0a5",
                "bad": "\ue002\ue00d\ue074\ue027\ue03b\ue04f\ue059\ue0d0\ue089",
                "bag": "\ue006\ue011\ue029\ue03c\ue051\ue059\ue084\ue0c6\ue0d9",
                "bakery": "\ue0e9\ue000\ue00c\ue029\ue03c\ue052\ue0e6\ue051\ue059\ue0d1\ue0e2\ue089\ue0c7\ue0aa\ue001"
                          "\ue00c\ue0e3\ue0d8",
                "ball": "\ue0e9\ue005\ue00e\ue0e6\ue005\ue011\ue00e\ue020\ue0e6\ue027\ue03c\ue0d0\ue04f\ue088\ue0c6"
                        "\ue0d8",
                "beautiful": "\ue006\ue010\ue020\ue038\ue04a\ue0d1\ue0e2\ue08c\ue0aa\ue005\ue00c\ue0e3\ue051",
                "cold": "\ue0e9\ue000\ue00d\ue031\ue03e\ue0e2\ue050\ue0e6\ue050\ue058\ue0e7\ue058\ue050\ue0e6\ue050"
                        "\ue0e3\ue0e2\ue086\ue0c6\ue0bb\ue0aa\ue027\ue03e\ue0e3",
                "beer": "\ue00a\ue020\ue03e\ue04a\ue08c\ue0d1\ue0a6"
                }
    for type, hamnosys in type2ham.items():
        with torch.no_grad():
            seq_iter = model.forward(text=hamnosys)
            for i in range(model.num_steps):
                seq = next(seq_iter)
            data = torch.unsqueeze(seq, 1).cpu()
            conf = torch.ones_like(data[:, :, :, 0])
            pose_body = NumPyPoseBody(25, data.numpy(), conf.numpy())
            predicted_pose = Pose(pose_header, pose_body)
            pose_hide_legs(predicted_pose)
            visualize_pose([predicted_pose], f"bsl_{type}.mp4", "videos/bsl_test")


    # first_pose = dataset[2]["pose"]["data"][0]
    # output_dir = f"/home/nlp/rotemsh/transcription/text_to_pose/videos/{experiment_name}/gsl_test"
    # os.makedirs(output_dir, exist_ok=True)
    # with torch.no_grad():
    #     for i, datum in enumerate(dataset):
    #         # if "gsl" in datum["id"]:
    #         seq_iter = model.forward(text=datum["text"], first_pose=first_pose)
    #         for i in range(model.num_steps):
    #             seq = next(seq_iter)
    #         visualize_seq(seq, pose_header, output_dir, datum["id"], datum["pose"]["obj"])

    # args.checkpoint = "/home/nlp/rotemsh/transcription/text_to_pose/models/20_steps/model-v2.ckpt"
    # args.pred_output = "/home/nlp/rotemsh/transcription/videos/20_steps"
    # args.ffmpeg_path = "/home/nlp/rotemsh/ffmpeg-5.0.1-amd64-static/"
    #
    # if args.checkpoint is None:
    #     raise Exception("Must specify `checkpoint`")
    # if args.pred_output is None:
    #     raise Exception("Must specify `pred_output`")
    # if args.ffmpeg_path is None:
    #     raise Exception("Must specify `ffmpeg_path`")
    #
    # os.makedirs(args.pred_output, exist_ok=True)
    #
    # dataset = get_dataset(name=args.dataset, poses=args.pose, fps=args.fps, components=args.pose_components,
    #                       max_seq_size=args.max_seq_size, split="train[:20]")
    #
    # _, num_pose_joints, num_pose_dims = dataset[0]["pose"]["data"].shape
    # pose_header = dataset.data[0]["pose"].header
    #
    # model_args = dict(tokenizer=HamNoSysTokenizer(),
    #                   pose_dims=(num_pose_joints, num_pose_dims),
    #                   hidden_dim=args.hidden_dim,
    #                   text_encoder_depth=args.text_encoder_depth,
    #                   pose_encoder_depth=args.pose_encoder_depth,
    #                   encoder_heads=args.encoder_heads,
    #                   max_seq_size=args.max_seq_size,
    #                   num_steps=args.num_steps)
    #
    # model = IterativeTextGuidedPoseGenerationModel.load_from_checkpoint(args.checkpoint, **model_args)
    # model.eval()
    #
    # html = []
    #
    # with torch.no_grad():
    #     for datum in dataset:
    #         first_pose = datum["pose"]["data"][0]
    #         # datum["text"] = ""
    #         seq_iter = model.forward(text=datum["text"], first_pose=first_pose)
    #         for i in range(10):  # This loop is near instantaneous
    #             seq = next(seq_iter)
    #
    #         data = torch.unsqueeze(seq, 1).cpu()
    #         conf = torch.ones_like(data[:, :, :, 0])
    #         pose_body = NumPyPoseBody(args.fps, data.numpy(), conf.numpy())
    #         predicted_pose = Pose(pose_header, pose_body)
    #         pose_hide_legs(predicted_pose)
    #
    #         html.append(visualize_poses(_id=datum["id"],
    #                                     text=datum["text"],
    #                                     poses=[datum["pose"]["obj"], predicted_pose]))

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

    # with open(os.path.join(args.pred_output, "index.html"), "w", encoding="utf-8") as f:
    #     f.write(
    #         "<style>@font-face {font-family: HamNoSys;src: url(HamNoSys.ttf);}.hamnosys{font-family: HamNoSys}</style>")
    #     f.write("<br><br><br>".join(html))
    #
    # shutil.copyfile(model.tokenizer.font_path, os.path.join(args.pred_output, "HamNoSys.ttf"))
