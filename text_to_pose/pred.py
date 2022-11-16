import os
from typing import List
import cv2
import numpy as np
import torch
import json
import random

from pose_format import Pose
from pose_format.numpy import NumPyPoseBody
from pose_format.pose_visualizer import PoseVisualizer

import sys
rootdir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, rootdir)

from shared.pose_utils import pose_normalization_info, pose_hide_legs
from text_to_pose.args import args
from text_to_pose.data import get_dataset
from text_to_pose.model import IterativeTextGuidedPoseGenerationModel
from text_to_pose.tokenizers.hamnosys.hamnosys_tokenizer import HamNoSysTokenizer


def predict_pose(model, datum, pose_header, model_for_seq_len=None, first_pose_seq=False):
    first_pose = datum["pose"]["data"][0]
    if model_for_seq_len is not None:
        seq_len = int(model_for_seq_len.encode_text([datum["text"]])[1].item())
    elif first_pose_seq:
        seq_len = int(model.encode_text([datum["text"]])[1].item())
    else:
        seq_len = -1

    if first_pose_seq:
        seq = torch.stack([first_pose] * seq_len, dim=0)
    else:
        seq_iter = model.forward(text=datum["text"], first_pose=first_pose.cuda(), sequence_length=seq_len)
        for j in range(model.num_steps):
            seq = next(seq_iter)

    data = torch.unsqueeze(seq, 1).cpu()
    conf = torch.ones_like(data[:, :, :, 0])
    pose_body = NumPyPoseBody(25, data.numpy(), conf.numpy())
    predicted_pose = Pose(pose_header, pose_body)
    pose_hide_legs(predicted_pose)
    normalization_info = pose_normalization_info(predicted_pose.header)
    predicted_pose = predicted_pose.normalize(normalization_info)
    predicted_pose.focus()
    return predicted_pose


def get_normalized_frames(poses):
    frames = []
    for i in range(len(poses)):
        # Normalize pose
        normalization_info = pose_normalization_info(poses[i].header)
        pose = poses[i].normalize(normalization_info, scale_factor=100)
        pose.focus()
        visualizer = PoseVisualizer(pose, thickness=2)
        pose_frames = list(visualizer.draw())
        frames.append(pose_frames)

    return frames


def visualize_pose(poses: List[Pose], pose_name: str, output_dir: str, slow: bool = False):
    os.makedirs(output_dir, exist_ok=True)
    f_name = os.path.join(output_dir, pose_name)
    fps = poses[0].body.fps
    if slow:
        f_name = f_name[:-4] + "_slow" + ".mp4"
        fps = poses[0].body.fps // 2

    frames = get_normalized_frames(poses)

    if len(poses) == 1:
        image_size = (frames[0][0].shape[1], frames[0][0].shape[0])
        out = cv2.VideoWriter(f_name, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, image_size)
        for frame in frames[0]:
            out.write(frame)
        out.release()
        return

    text_margin = 50
    image_size = (max(frames[0][0].shape[1], frames[1][0].shape[1]) * 2,
                  max(frames[0][0].shape[0], frames[1][0].shape[0]) + text_margin)
    out = cv2.VideoWriter(f_name, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, image_size)
    for i, frame in enumerate(frames[1]):
        if len(frames[0]) > i:
            empty = np.full((image_size[1], image_size[0]//2, 3), 255, dtype=np.uint8)
            empty[text_margin:frames[0][i].shape[0]+text_margin, :frames[0][i].shape[1]] = frames[0][i]
        label_frame = empty
        pred_frame = np.full((image_size[1], image_size[0]//2, 3), 255, dtype=np.uint8)
        pred_frame[text_margin:frame.shape[0]+text_margin, :frame.shape[1]] = frame
        label_pred_im = concat_and_add_label(label_frame, pred_frame, image_size)
        out.write(label_pred_im)

    if i < len(frames[0])-1:
        for frame in frames[0][i:]:
            label_frame = np.full((image_size[1], image_size[0] // 2, 3), 255, dtype=np.uint8)
            label_frame[text_margin:frame.shape[0] + text_margin, :frame.shape[1]] = frame
            # pred_frame = np.full((image_size[1], image_size[0] // 2, 3), 255, dtype=np.uint8)
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


def create_ref2poses_video(poses: List[Pose], output_dir, vid_name):
    os.makedirs(output_dir, exist_ok=True)
    f_name = os.path.join(output_dir, vid_name)
    fps = poses[0].body.fps
    frames = get_normalized_frames(poses)
    margins = 20
    max_len = max([len(pose_frames) for pose_frames in frames])
    shape_0 = max([pose_frames[0].shape[0] for pose_frames in frames]) + margins
    shape_1 = sum([pose_frames[0].shape[1] for pose_frames in frames]) + margins
    image_size = (shape_1, shape_0)
    out = cv2.VideoWriter(f_name, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, image_size)
    for i in range(max_len):
        idx = 0
        frame = np.full((image_size[1], image_size[0], 3), 255, dtype=np.uint8)
        for pose_frames in frames:
            if i < len(pose_frames):
                frame[margins//2:margins//2+pose_frames[i].shape[0], idx:idx+pose_frames[i].shape[1]] = pose_frames[i]
            idx += pose_frames[0].shape[1]
        cv2.line(frame, (frames[0][0].shape[1] + 3, 0), (frames[0][0].shape[1] + 3, len(frame)), 0)
        out.write(frame)
    out.release()


def visualize_seq(seq, pose_header, output_dir, id, label_pose=None, fps=25):
    with torch.no_grad():
        data = torch.unsqueeze(seq, 1).cpu()
        conf = torch.ones_like(data[:, :, :, 0])
        pose_body = NumPyPoseBody(fps, data.numpy(), conf.numpy())
        predicted_pose = Pose(pose_header, pose_body)
        if "pose_keypoints_2d" in [c.name for c in pose_header.components]:
            pose_hide_legs(predicted_pose)

        pose_name = f"{id}.mp4"
        if label_pose is None:
            visualize_pose([predicted_pose], pose_name, output_dir)
        else:
            visualize_pose([label_pose, predicted_pose], pose_name, output_dir)


def visualize_sequences(sequences, pose_header, output_dir, id, label_pose=None, fps=25, labels=None):
    os.makedirs(output_dir, exist_ok=True)
    poses = [label_pose]
    for seq in sequences:
        data = torch.unsqueeze(seq, 1).cpu()
        conf = torch.ones_like(data[:, :, :, 0])
        pose_body = NumPyPoseBody(fps, data.numpy(), conf.numpy())
        pose = Pose(pose_header, pose_body)
        pose_hide_legs(pose)
        poses.append(pose)

    pose_name = f"{id}_merged.mp4"
    font = cv2.FONT_HERSHEY_TRIPLEX
    color = (0, 0, 0)
    f_name = os.path.join(output_dir, pose_name)
    all_frames = get_normalized_frames(poses)
    if labels is None:
        labels = ["ground truth", "step 10", "step 8", "step 7", "step 6", "last step"]
    text_margin = 50
    w = max([frames[0].shape[1] for frames in all_frames])
    h = max([frames[0].shape[0] for frames in all_frames])
    image_size = (w * len(poses), h + text_margin)
    out = cv2.VideoWriter(f_name, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, image_size)
    max_len = max([len(frames) for frames in all_frames])

    for i in range(max_len+1):
        all_video_frames = []
        for j, frames in enumerate(all_frames):
            if len(frames) > i:
                cur_frame = np.full((image_size[1], image_size[0] // len(poses), 3), 255, dtype=np.uint8)
                cur_frame[text_margin:frames[i].shape[0] + text_margin, :frames[i].shape[1]] = frames[i]
                cur_frame = cv2.putText(cur_frame, labels[j], (5, 20), font, 0.5, color, 1, 0)
            else:
                cur_frame = prev_video_frames[j]
            all_video_frames.append(cur_frame)
        merged_frame = np.concatenate(all_video_frames, axis=1)
        out.write(merged_frame)
        prev_video_frames = all_video_frames

    out.release()


def vis_label_only(dataset, output_dir, skip_existing=False):
    os.makedirs(output_dir, exist_ok=True)
    for i, datum in enumerate(dataset):
        pose_name = f"{datum['id']}.mp4"
        if skip_existing and os.path.isfile(os.path.join(output_dir, pose_name)):
            continue
        label_pose = datum["pose"]["obj"]
        visualize_pose([label_pose], pose_name, output_dir)


def pred(model, dataset, output_dir, vis_process=False, vis_pred_only=False, gen_k=30, vis=True, subset=None,
         model_for_seq_len=None):

    os.makedirs(output_dir, exist_ok=True)
    _, num_pose_joints, num_pose_dims = dataset[0]["pose"]["data"].shape
    pose_header = dataset[0]["pose"]["obj"].header
    preds = []

    model.eval()
    with torch.no_grad():
        for i, datum in enumerate(dataset):
            if subset is not None and datum["id"] not in subset:
                continue
            if i > gen_k and subset is None:
                break
            first_pose = datum["pose"]["data"][0]
            seq_len = -1
            sequences = []
            if model_for_seq_len is not None:
                seq_len = int(model_for_seq_len.encode_text([datum["text"]])[1].item())

            seq_iter = model.forward(text=datum["text"], first_pose=first_pose, sequence_length=seq_len)
            for j in range(model.num_steps):
                seq = next(seq_iter)
                if vis_process and j in [0, 2, 3, 4, 9]:
                    sequences.append(seq)

            if vis_process:
                visualize_sequences(sequences, pose_header, output_dir, datum["id"], datum["pose"]["obj"])
            if vis and vis_pred_only:
                visualize_seq(seq, pose_header, output_dir, datum["id"], label_pose=None)
            elif vis:
                visualize_seq(seq, pose_header, output_dir, datum["id"], datum["pose"]["obj"])
            else:
                data = torch.unsqueeze(seq, 1).cpu()
                conf = torch.ones_like(data[:, :, :, 0])
                pose_body = NumPyPoseBody(25, data.numpy(), conf.numpy())
                predicted_pose = Pose(pose_header, pose_body)
                pose_hide_legs(predicted_pose)
                preds.append(predicted_pose)
    return preds


def add_orig_vid_to_model_vid(orig_vids_path, model_vids_path, output_dir="", add_title=True):
    if output_dir == "":
        output_dir = os.path.join(model_vids_path, "combined")
    os.makedirs(output_dir, exist_ok=True)
    for vid in os.listdir(model_vids_path):
        if not vid.endswith(".mp4"):
            continue
        if vid in output_dir:
            continue
        cap_orig = cv2.VideoCapture(os.path.join(orig_vids_path, vid))
        cap_keypoints = cv2.VideoCapture(os.path.join(model_vids_path, vid))
        ret, frame_orig = cap_orig.read()
        if "pjm" in vid:
            if vid == "pjm_2792.mp4":
                for i in range(32):
                    _, frame_orig = cap_orig.read()
            else:
                for i in range(48):
                    _, frame_orig = cap_orig.read()
                if vid == "pjm_1471.mp4":
                    for i in range(4):
                        _, frame_orig = cap_orig.read()
                if vid in ["pjm_3144.mp4", "pjm_572.mp4"]:
                    for i in range(10):
                        _, frame_orig = cap_orig.read()
        _, frame_keypoints = cap_keypoints.read()
        if frame_orig is None or frame_keypoints is None:
            print(vid)
            continue
        scale = False
        if frame_orig.shape[0] > 1.5 * frame_keypoints.shape[0]:
            scale_factor = frame_orig.shape[0] / frame_keypoints.shape[0]
            frame_orig = cv2.resize(frame_orig, (int(frame_orig.shape[1] // scale_factor),
                                                 int(frame_orig.shape[0] // scale_factor)))
            scale = True
        orig_height = frame_orig.shape[0]
        orig_width = frame_orig.shape[1]
        scaled_orig_width = orig_width
        if orig_width > 400:
            scaled_orig_width -= 300
        elif "gsl" in vid or vid in ["VILLA_.mp4", "_NUM-QUINZE.mp4"]:
            scaled_orig_width -= 100
        height = max(orig_height, frame_keypoints.shape[0])
        out_vid = cv2.VideoWriter(os.path.join(model_vids_path, "combined", vid),
                                  cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 25,
                                  (scaled_orig_width + frame_keypoints.shape[1], height))
        i = 1
        while cap_orig.isOpened():
            if add_title:
                frame_keypoints[:40] = np.full((40, frame_keypoints.shape[1], 3), 255)
                label_coord = frame_keypoints.shape[1] // 5 - 22
                pred_coord = int(3.2 * frame_keypoints.shape[1] // 5)
                if vid == "17391.mp4":
                    label_coord -= 30
                    pred_coord += 5
                elif vid == "11127.mp4":
                    label_coord -= 3
                elif vid in ["13074.mp4", "pjm_154.mp4"]:
                    label_coord -= 18
                    pred_coord += 8
                elif vid in ["19760.mp4", "69724.mp4"]:
                    label_coord += 5
                elif vid == "pjm_1471.mp4":
                    pred_coord -= 7
                    label_coord += 5
                elif vid == "pjm_3144.mp4":
                    pred_coord -= 5
                    label_coord -= 5

                title_coord = 25
                if vid in ["8633.mp4", "pjm_2792.mp4"]:
                    title_coord = 40

                frame_keypoints = cv2.putText(frame_keypoints, "ground truth", (label_coord, title_coord),
                                          cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 0, 0), 1, 0)
                frame_keypoints = cv2.putText(frame_keypoints, "prediction", (pred_coord, title_coord),
                                          cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 0, 0), 1, 0)

            if frame_orig.shape[1] > 400:
                frame_orig = frame_orig[:, 150:-150]
            elif ret and ("gsl" in vid or vid in ["VILLA_.mp4", "_NUM-QUINZE.mp4"]):
                frame_orig = frame_orig[:, 50:-50]
            if ret and vid in ["pjm_1471.mp4", "pjm_572.mp4", "pjm_1561.mp4", "pjm_1634.mp4", "pjm_1944"]:
                frame_orig = np.fliplr(frame_orig)
            if height == orig_height:
                empty = np.full((height, frame_keypoints.shape[1], 3), 255, dtype=np.uint8)
                empty[-frame_keypoints.shape[0]:, :] = frame_keypoints
                cat = np.concatenate((frame_orig, empty), axis=1)
            else:
                empty = np.full((height, scaled_orig_width, 3), 255, dtype=np.uint8)
                empty[(height-orig_height)//2:-(height-orig_height)//2, :] = frame_orig
                cat = np.concatenate((empty, frame_keypoints), axis=1)
            out_vid.write(cat)
            prev_frame_orig = frame_orig
            prev_frame_keypoints = frame_keypoints
            ret, frame_orig = cap_orig.read()
            ret1, frame_keypoints = cap_keypoints.read()
            if not ret and not ret1:
                break
            elif not ret:
                frame_orig = prev_frame_orig #np.full((orig_height, orig_width, 3), 255, dtype=np.uint8)
            elif not ret1:
                frame_keypoints = prev_frame_keypoints #np.full(keypoints_shape, 255, dtype=np.uint8)
            if scale:# and ret:
                frame_orig = cv2.resize(frame_orig, (orig_width, orig_height))
            i += 1
        out_vid.release()


def generate_keypoints_file_from_pose(pose, output_dir, pose_header):
    os.makedirs(output_dir, exist_ok=True)
    data = torch.unsqueeze(pose, 1).cpu()
    conf = torch.ones_like(data[:, :, :, 0])
    pose_body = NumPyPoseBody(25, data.numpy(), conf.numpy())
    predicted_pose = Pose(pose_header, pose_body)
    pose_hide_legs(predicted_pose)
    normalization_info = pose_normalization_info(pose_header)
    pose = predicted_pose.normalize(normalization_info, scale_factor=100)
    pose.focus()
    predicted_pose_data = pose.body.data.squeeze(1)
    for i, frame in enumerate(predicted_pose_data):
        pose_keypoints = []
        for j in range(25):
            pose_keypoints += [float(frame[j][0]), float(frame[j][1]), 1.0]
        face_keypoints = []
        for j in range(25, 95):
            face_keypoints += [float(frame[j][0]), float(frame[j][1]), 1.0]
        lhand_keypoints = []
        for j in range(95, 116):
            lhand_keypoints += [float(frame[j][0]), float(frame[j][1]), 1.0]
        rhand_keypoints = []
        for j in range(116, 137):
            rhand_keypoints += [float(frame[j][0]), float(frame[j][1]), 1.0]

        keypoints_json = {"people": [{"person_id": [-1], "pose_keypoints_2d": pose_keypoints,
                                      "face_keypoints_2d": face_keypoints,
                                      "hand_left_keypoints_2d": lhand_keypoints,
                                      "hand_right_keypoints_2d": rhand_keypoints}]}

        id = output_dir[output_dir.rfind(os.path.sep)+1:]
        num_frame = '%012d' % i
        output_path = os.path.join(output_dir, f"{id}_{num_frame}_keypoints.json")
        with open(output_path, 'w') as f:
            json.dump(keypoints_json, f)


def generate_different_step_num_vids(ten_steps_model, model_args, dataset, subset, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    _, num_pose_joints, num_pose_dims = dataset[0]["pose"]["data"].shape
    pose_header = dataset[0]["pose"]["obj"].header

    num_steps = [5, 20]
    models = {10: ten_steps_model}
    for step_num in num_steps:
        ckpt = f"/home/nlp/rotemsh/transcription/models/exclude_sep_{step_num}_steps/model.ckpt"
        model_args["num_steps"] = step_num
        model = IterativeTextGuidedPoseGenerationModel.load_from_checkpoint(ckpt, **model_args)
        model.eval()
        models[step_num] = model

    with torch.no_grad():
        for datum in dataset:
            if datum["id"] in subset:
                first_pose = datum["pose"]["data"][0]
                sequences = []
                seq_len = int(ten_steps_model.encode_text([datum["text"]])[1].item())
                for step_num, model in sorted(models.items()):
                    seq_iter = model.forward(text=datum["text"], first_pose=first_pose, sequence_length=seq_len)
                    for j in range(model.num_steps):
                        seq = next(seq_iter)
                    sequences.append(seq)

                labels = ["ground truth", "5 steps", "10 steps", "20 steps"]
                visualize_sequences(sequences, pose_header, output_dir, datum["id"], label_pose=datum["pose"]["obj"],
                                    labels=labels)


if __name__ == '__main__':
    # orig_vids_path = r"/home/nlp/rotemsh/transcription/text_to_pose/results/videos"
    # model_vids_path = r"/home/nlp/rotemsh/transcription/text_to_pose/results/pose_videos"
    # add_orig_vid_to_model_vid(orig_vids_path, model_vids_path)
    # exit()

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    DATASET_SIZE = 5754
    test_size = int(0.1 * DATASET_SIZE)
    print("test size", test_size)
    test_split = f'test[:{test_size}]'
    # 14 - 10334
    # 18 - pjm_1220
    # 91 - 2358
    # 246 - 2507
    # 369 - 84347

    dataset = get_dataset(name=args.dataset, poses=args.pose, fps=args.fps,
                          components=args.pose_components, exclude=True,
                          max_seq_size=args.max_seq_size, split=test_split)

    experiment_name = "reproduce_exclude_sep_2"
    _, num_pose_joints, num_pose_dims = dataset[0]["pose"]["data"].shape

    model_args = dict(tokenizer=HamNoSysTokenizer(),
                      pose_dims=(num_pose_joints, num_pose_dims),
                      hidden_dim=args.hidden_dim,
                      text_encoder_depth=args.text_encoder_depth,
                      pose_encoder_depth=args.pose_encoder_depth,
                      encoder_heads=args.encoder_heads,
                      max_seq_size=args.max_seq_size,
                      num_steps=args.num_steps,
                      tf_p=args.tf_p,
                      masked_loss=args.masked_loss,
                      separate_positional_embedding=args.separate_positional_embedding,
                      num_pose_projection_layers=args.num_pose_projection_layers,
                      concat=True
                      )

    args.checkpoint = f"/home/nlp/rotemsh/transcription/models/{experiment_name}/model.ckpt"
    model = IterativeTextGuidedPoseGenerationModel.load_from_checkpoint(args.checkpoint, **model_args)
    model.eval()

    output_dir = f"/home/nlp/rotemsh/transcription/text_to_pose/results/process"
    # subset = ["51119", "84347"]
    # generate_different_step_num_vids(model, model_args, dataset, subset, output_dir)
    # exit()

    model_for_seq_len = None
    if args.num_steps != 10:
        checkpoint = f"/home/nlp/rotemsh/transcription/models/reproduce_exclude_sep_2/model.ckpt"
        model_args["num_steps"] = 10
        model_for_seq_len = IterativeTextGuidedPoseGenerationModel.load_from_checkpoint(checkpoint,
                                                                                        **model_args)
        model_for_seq_len.eval()

    # output_dir = f"/home/nlp/rotemsh/transcription/text_to_pose/videos/" \
    #              f"{experiment_name}/all_test"
    subset = ["84347", "pjm_1675"]
    pred(model, dataset, output_dir, gen_k=600, subset=subset, vis_process=True)#, model_for_seq_len=model_for_seq_len)
    exit()

    # pose_header = dataset.data[0]["pose"].header
    #
    # # idx2id = {14: "10334", 18: "pjm_1220", 91: "2358", 246: "2507", 369: "84347", 55: "9615", 59: "27007",
    # # 217: "8633", 265: "58978", 246: "2507", 18: "pjm_1220"}
    # # idx = 18
    # with torch.no_grad():
    #     for datum in dataset:
    #         if datum["id"] in subset:
    #             # for idx, id in idx2id.items():
    #             text = datum["text"]
    #             # #dataset[0]["text"]
    #             first_pose = datum["pose"]["data"][0] #dataset[idx]["pose"]["data"][0]
    #             seq_iter = model.forward(text=text, first_pose=first_pose)
    #             for j in range(model.num_steps):
    #                 seq = next(seq_iter)
    #             visualize_seq(seq, pose_header, output_dir, f"{datum['id']}_large", #_from_{idx2id[idx]}",
    #                           label_pose=datum["pose"]["obj"])