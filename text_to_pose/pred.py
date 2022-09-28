import os
import shutil
from typing import List
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import itertools
import math
import json
import pickle

from pose_format import Pose
from pose_format.numpy import NumPyPoseBody
from pose_format.pose_visualizer import PoseVisualizer

import sys
sys.path.insert(0, '/home/nlp/rotemsh/transcription')

from shared.pose_utils import pose_normalization_info, pose_hide_legs, get_original_pose
from text_to_pose.args import args
from text_to_pose.data import get_dataset
# from text_to_pose.tokenizers.hamnosys.hamnosys_tokenizer import HamNoSysTokenizer

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Only use CPU


def draw_frame(frame, frame_confidence, img, pose, thickness=None, line_thickness=None, add_gaussians=False):
    background_color = img[0][0]  # Estimation of background color for opacity. `mean` is slow
    thickness = thickness if thickness is not None else \
        round(math.sqrt(img.shape[0] * img.shape[1]) / 150)
    radius = round(thickness / 2)

    for person, person_confidence in zip(frame, frame_confidence):
        c = person_confidence.tolist()
        idx = 0
        for component in pose.header.components:
            colors = [np.array(c[::-1]) for c in component.colors]

            def _point_color(p_i: int):
                opacity = c[p_i + idx]
                np_color = colors[p_i % len(component.colors)] * opacity + (1 - opacity) * background_color
                return tuple([int(c) for c in np_color])

            # Draw Points
            for i, point_name in enumerate(component.points):
                if c[i + idx] > 0:
                    cv2.circle(img=img, center=tuple(person[i + idx][:2]), radius=radius,
                               color=_point_color(i), thickness=thickness, lineType=16)
                    if add_gaussians:
                        x = person[i + idx][0]
                        y = person[i + idx][1]
                        h = w = 12
                        frame_h, frame_w, _ = img.shape
                        y1 = max(y - h // 2, 0)
                        y2 = min(y + h // 2, frame_h)
                        x1 = max(x - w // 2, 0)
                        x2 = min(x + w // 2, frame_w)
                        img[y1:y2, x1:x2] = cv2.GaussianBlur(img[y1:y2, x1:x2], (7, 7), 5)

            if pose.header.is_bbox:
                point1 = tuple(person[0 + idx].tolist())
                point2 = tuple(person[1 + idx].tolist())
                color = tuple(np.mean([_point_color(0), _point_color(1)], axis=0))
                cv2.rectangle(img=img, pt1=point1, pt2=point2, color=color, thickness=thickness)
            else:
                int_person = person.astype(np.int32)
                if line_thickness is not None:
                    # Draw Limbs
                    for (p1, p2) in component.limbs:
                        if c[p1 + idx] > 0 and c[p2 + idx] > 0:
                            point1 = tuple(int_person[p1 + idx].tolist()[:2])
                            point2 = tuple(int_person[p2 + idx].tolist()[:2])
                            color = tuple(np.mean([_point_color(p1), _point_color(p2)], axis=0))
                            cv2.line(img, point1, point2, color, 2, lineType=cv2.LINE_AA)
            idx += len(component.points)
    return img


def draw_pose_on_video(video_path, pose):
    int_data = np.array(np.around(pose.body.data.data), dtype="int32")
    max_frames = len(int_data)

    def get_frames(video_path):
        cap = cv2.VideoCapture(video_path)
        while True:
            ret, vf = cap.read()
            if not ret:
                break
            yield vf
        cap.release()

    background_video = iter(get_frames(video_path))
    for frame, confidence, background in itertools.islice(zip(int_data, pose.body.confidence, background_video),
                                                          max_frames):
        yield draw_frame(frame, confidence, background, pose, thickness=2)


def get_video_with_pose_gaussians(video_path, pose_path, confidence_threshold=0.2):  # TODO- my new method

    def get_frames(video_path, pose_path):
        cap = cv2.VideoCapture(video_path)
        i = 0
        frames = []
        while True:
            ret, vf = cap.read()
            if not ret:
                break
            vid_name = pose_path[pose_path.rfind(os.path.sep)+1:]
            frame_num = '%012d' % i
            frame_json_name = os.path.join(pose_path, f"{vid_name}_{frame_num}_keypoints.json")
            with open(frame_json_name, 'r') as f:
                frame_pose_json = json.load(f)["people"][0]
                frame_pose = np.concatenate((frame_pose_json["pose_keypoints_2d"], frame_pose_json[
                    "hand_left_keypoints_2d"], frame_pose_json["hand_right_keypoints_2d"], frame_pose_json[
                    "face_keypoints_2d"]))
            frames.append((vf, frame_pose))
            i += 1
        cap.release()
        return frames

    h = w = 12
    background_video_and_pose = get_frames(video_path, pose_path)
    frames = []
    for frame, pose in background_video_and_pose:
        frame_h, frame_w, _ = frame.shape
        for i in range(0, len(pose), 3):
            if pose[i+2] > confidence_threshold:
                x = int(pose[i])
                y = int(pose[i+1])
                if x > frame_w or y > frame_h:
                    continue
                y1 = max(y - h // 2, 0)
                y2 = min(y + h // 2, frame_h)
                x1 = max(x - w // 2, 0)
                x2 = min(x + w // 2, frame_w)
                frame[y1:y2, x1:x2] = cv2.GaussianBlur(frame[y1:y2, x1:x2], (7, 7), 5)
        frames.append(frame)
    return frames


def get_pose_gaussians(video_path, pose_path, confidence_threshold=0.2):  # TODO- my new method

    def get_frames(video_path, pose_path):
        cap = cv2.VideoCapture(video_path)
        ret, vf = cap.read()
        pose_files = os.listdir(pose_path)
        frames = []
        for pose_file in pose_files:
            frame_json_name = os.path.join(pose_path, pose_file)
            with open(frame_json_name, 'r') as f:
                frame_pose_json = json.load(f)["people"][0]
                frame_pose = np.concatenate((frame_pose_json["pose_keypoints_2d"], frame_pose_json[
                    "hand_left_keypoints_2d"], frame_pose_json["hand_right_keypoints_2d"], frame_pose_json[
                                                 "face_keypoints_2d"]))
            frames.append((frame_pose, np.ones_like(vf)*255))
        cap.release()
        return frames

    h = w = 12
    thickness = 2
    radius = round(thickness / 2)
    frames = get_frames(video_path, pose_path)
    new_frames = []
    for pose, bg in frames:
        frame_h, frame_w, _ = bg.shape
        for i in range(0, len(pose), 3):
            if pose[i + 2] > confidence_threshold:
                x = int(pose[i])
                y = int(pose[i + 1])
                if x > frame_w or y > frame_h:
                    continue
                cv2.circle(img=bg, center=(x, y), radius=radius,
                           color=(255, 0, 0), thickness=thickness, lineType=16)
                y1 = max(y - h // 2, 0)
                y2 = min(y + h // 2, frame_h)
                x1 = max(x - w // 2, 0)
                x2 = min(x + w // 2, frame_w)
                bg[y1:y2, x1:x2] = cv2.GaussianBlur(bg[y1:y2, x1:x2], (7, 7), 5)
        new_frames.append(bg)
    return np.array(new_frames)


def get_normalized_frames(poses, add_gaussians=False):
    frames = []
    for i in range(len(poses)):
        # Normalize pose
        normalization_info = pose_normalization_info(poses[i].header)
        pose = poses[i].normalize(normalization_info, scale_factor=100)
        pose.focus()
        if add_gaussians:
            pose_frames = []
            background_color = (255, 255, 255)
            int_data = np.array(np.around(pose.body.data.data), dtype="int32")
            background = np.full((pose.header.dimensions.height, pose.header.dimensions.width, 3),
                                 fill_value=background_color, dtype="uint8")
            for frame, confidence in zip(int_data, poses[i].body.confidence):
                pose_frames.append(draw_frame(frame, confidence, img=background.copy(), pose=pose,
                                              thickness=2, add_gaussians=add_gaussians))

        else:
            visualizer = PoseVisualizer(pose, thickness=2)
            pose_frames = list(visualizer.draw())

        frames.append(pose_frames)

    return frames


def visualize_pose(poses: List[Pose], pose_name: str, output_dir: str, slow: bool = False,
                   add_gaussians: bool = False, is_relative_pose: bool = False):
    os.makedirs(output_dir, exist_ok=True)
    f_name = os.path.join(output_dir, pose_name)
    fps = poses[0].body.fps
    if slow:
        f_name = f_name[:-4] + "_slow" + ".mp4"
        fps = poses[0].body.fps // 2
    
    if is_relative_pose:
        get_original_pose(poses)
    frames = get_normalized_frames(poses, add_gaussians)

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
    out = cv2.VideoWriter(f_name, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, image_size)
    for i, frame in enumerate(frames[1]):
        empty = np.full((image_size[1], image_size[0]//2, 3), 255, dtype=np.uint8)
        if len(frames[0]) > i:
            empty[40:frames[0][i].shape[0]+40, :frames[0][i].shape[1]] = frames[0][i]
        label_frame = empty
        pred_frame = np.full((image_size[1], image_size[0]//2, 3), 255, dtype=np.uint8)
        pred_frame[40:frame.shape[0]+40, :frame.shape[1]] = frame
        label_pred_im = concat_and_add_label(label_frame, pred_frame, image_size)
        out.write(label_pred_im)

    if i < len(frames[0])-1:
        for frame in frames[0][i:]:
            label_frame = np.full((image_size[1], image_size[0] // 2, 3), 255, dtype=np.uint8)
            label_frame[40:frame.shape[0] + 40, :frame.shape[1]] = frame
            pred_frame = np.full((image_size[1], image_size[0] // 2, 3), 255, dtype=np.uint8)
            label_pred_im = concat_and_add_label(label_frame, pred_frame, image_size)
            out.write(label_pred_im)

    out.release()


def visualize_keypoints_as_gaussians(pose):
    int_data = np.array(np.around(pose.body.data.data), dtype="int32")
    background = np.full((pose.header.dimensions.height, pose.header.dimensions.width, 3),
                         fill_value=255, dtype="uint8")
    h = w = 12
    frame_h, frame_w = pose.header.dimensions.height, pose.header.dimensions.width
    # keypoints = set([tuple(res[:-1]) for res in np.argwhere(frames != np.array([255]))])
    frames = []
    for frame in pose.body.data:
        for x, y in frame[0]:
            if np.ma.is_masked(frame[0][x, y]):
                continue
            y1 = max(y-h//2, 0)
            y2 = min(y+h//2, frame_h)
            x1 = max(x - w // 2, 0)
            x2 = min(x + w // 2, frame_w)
            frame[0][y1:y2, x1:x2] = cv2.GaussianBlur(frame[0][y1:y2, x1:x2], (7, 7), 5)
            plt.imshow(frame[0])
            plt.show()
        frames.append(frame[0])
    return frames


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


def visualize_poses(_id: str, text: str, poses: List[Pose], add_gaussians: bool = False) -> str:
    lengths = " / ".join([str(len(p.body.data)) for p in poses])
    html_tags = f"<h3><u>{_id}</u>: <span class='hamnosys'>{text}</span> ({lengths})</h3>"

    pose_name = f"{_id}.mp4"
    visualize_pose(poses, pose_name, args.pred_output, add_gaussians=add_gaussians)
    html_tags += f"<video src='{pose_name}' controls preload='none'></video>"

    return html_tags


def visualize_seq(seq, pose_header, output_dir, id, label_pose=None, fps=25, add_gaussians=False):
    with torch.no_grad():
        data = torch.unsqueeze(seq, 1).cpu()
        conf = torch.ones_like(data[:, :, :, 0])
        pose_body = NumPyPoseBody(fps, data.numpy(), conf.numpy())
        predicted_pose = Pose(pose_header, pose_body)
        if "pose_keypoints_2d" in [c.name for c in pose_header.components]:
            pose_hide_legs(predicted_pose)

        pose_name = f"{id}.mp4"
        if label_pose is None:
            visualize_pose([predicted_pose], pose_name, output_dir, add_gaussians=add_gaussians)
        else:
            visualize_pose([label_pose, predicted_pose], pose_name, output_dir, add_gaussians=add_gaussians)


def vis_label_only(dataset, output_dir, skip_existing=False, add_gaussians=False, only_dgs=False,
                   is_relative_pose=False):
    os.makedirs(output_dir, exist_ok=True)
    # _, num_pose_joints, num_pose_dims = dataset[0]["pose"]["data"].shape
    for i, datum in enumerate(dataset):
        if only_dgs and not datum['id'].isnumeric():
            continue
        pose_name = f"{datum['id']}.mp4"
        if skip_existing and os.path.isfile(os.path.join(output_dir, pose_name)):
            continue
        label_pose = datum["pose"]["obj"]
        visualize_pose([label_pose], pose_name, output_dir, add_gaussians=add_gaussians,
                       is_relative_pose=is_relative_pose)


def pred(model, dataset, output_dir, use_learned_first_pose=False, vis_process=False, add_gaussians=False,
         vis_pred_only=False, gen_k=30):
    os.makedirs(output_dir, exist_ok=True)
    _, num_pose_joints, num_pose_dims = dataset[0]["pose"]["data"].shape
    pose_header = dataset.data[0]["pose"].header
    first_pose = None

    model.eval()
    with torch.no_grad():
        for i, datum in enumerate(dataset):
            if i > gen_k:
                break
            if not use_learned_first_pose:
                first_pose = datum["pose"]["data"][0]
            seq_len = int(datum["pose"]["length"].item())  # TODO- change to trained model pred
            seq_iter = model.forward(text=datum["text"], first_pose=first_pose, sequence_length=seq_len)
            for j in range(model.num_steps):
                seq = next(seq_iter)
                if vis_process and j in [2, 4, 6, 8]:
                    visualize_seq(seq, pose_header, output_dir, f"{datum['id']}_step_{j}", datum["pose"]["obj"],
                                  add_gaussians)
            if vis_pred_only:
                visualize_seq(seq, pose_header, output_dir, datum["id"], label_pose=None, add_gaussians=add_gaussians)
            else:
                visualize_seq(seq, pose_header, output_dir, datum["id"], datum["pose"]["obj"], add_gaussians=add_gaussians)


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
        orig_height = frame_orig.shape[0]
        orig_width = frame_orig.shape[1]
        keypoints_shape = frame_keypoints.shape
        height = max(orig_height, frame_keypoints.shape[0])
        out_vid = cv2.VideoWriter(os.path.join(model_vids_path, "combined", vid),
                                  cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 25,
                                  (orig_width + frame_keypoints.shape[1], height))
        i = 1
        while cap_orig.isOpened():
            if height == orig_height:
                empty = np.full((height, frame_keypoints.shape[1], 3), 255, dtype=np.uint8)
                empty[-frame_keypoints.shape[0]:, :] = frame_keypoints
                cat = np.concatenate((frame_orig, empty), axis=1)
            else:
                empty = np.full((height, orig_width, 3), 255, dtype=np.uint8)
                empty[-orig_height:, :] = frame_orig
                cat = np.concatenate((empty, frame_keypoints), axis=1)
            out_vid.write(cat)
            ret, frame_orig = cap_orig.read()
            ret1, frame_keypoints = cap_keypoints.read()
            if not ret and not ret1:
                break
            elif not ret:
                frame_orig = np.zeros((orig_height, orig_width, 3), dtype=np.uint8)
            elif not ret1:
                frame_keypoints = np.zeros(keypoints_shape, dtype=np.uint8)
            if scale and ret:
                frame_orig = cv2.resize(frame_orig, (orig_width, orig_height))
            i += 1
        out_vid.release()


def generate_progress_vid(filename, vids_path):
    cap_orig = cv2.VideoCapture(os.path.join(vids_path, filename+".mp4"))
    pred_caps = [cv2.VideoCapture(os.path.join(vids_path, filename+f"_{i}.mp4")) for i in range(3)]

    _, frame_orig = cap_orig.read()
    _, frame_pred_0 = pred_caps[0].read()
    _, frame_pred_1 = pred_caps[1].read()
    _, frame_pred_2 = pred_caps[2].read()
    scale = False
    if frame_orig.shape[0] > 1.5 * frame_pred_0.shape[0]:
        scale_factor = frame_orig.shape[0] // frame_pred_0.shape[0]
        frame_orig = cv2.resize(frame_orig, (frame_orig.shape[1] // scale_factor,
                                             frame_orig.shape[0] // scale_factor))
        scale = True
    orig_height = frame_orig.shape[0]
    orig_width = frame_orig.shape[1]
    max_pred_height = max(frame_pred_0.shape[0], frame_pred_1.shape[0], frame_pred_2.shape[0])
    max_pred_width = max(frame_pred_0.shape[1], frame_pred_1.shape[1], frame_pred_2.shape[1])
    keypoints_shape = (max_pred_height, max_pred_width)
    height = max(orig_height, max_pred_height)
    out_vid = cv2.VideoWriter(os.path.join(vids_path, filename+"_combined.mp4"),
                              cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 25,
                              (orig_width + max_pred_width + 2*(max_pred_width//2), height))

    i = 1

    while cap_orig.isOpened():
        if frame_pred_0.shape[1] < max_pred_width:
            empty = np.full((frame_pred_0.shape[0], max_pred_width, 3), 255, dtype=np.uint8)
            empty[:, -frame_pred_0.shape[1]:] = frame_pred_0
            frame_pred_0 = empty
        if frame_pred_1.shape[1] < max_pred_width:
            empty = np.full((frame_pred_1.shape[0], max_pred_width, 3), 255, dtype=np.uint8)
            empty[:, -frame_pred_1.shape[1]:] = frame_pred_1
            frame_pred_1 = empty
        if frame_pred_2.shape[1] < max_pred_width:
            empty = np.full((frame_pred_2.shape[0], max_pred_width, 3), 255, dtype=np.uint8)
            empty[:, -frame_pred_2.shape[1]:] = frame_pred_2
            frame_pred_2 = empty
        frame_pred_1_cut = frame_pred_1[:, -frame_pred_1.shape[1]//2:]
        frame_pred_2_cut = frame_pred_2[:, -frame_pred_2.shape[1]//2:]
        if frame_pred_0.shape[0] < max_pred_height:
            empty = np.full((max_pred_height, frame_pred_0.shape[1], 3), 255, dtype=np.uint8)
            empty[-frame_pred_0.shape[0]:] = frame_pred_0
            frame_pred_0 = empty
        if frame_pred_1_cut.shape[0] < max_pred_height:
            empty = np.full((max_pred_height, frame_pred_1_cut.shape[1], 3), 255, dtype=np.uint8)
            empty[-frame_pred_1_cut.shape[0]:] = frame_pred_1_cut
            frame_pred_1_cut = empty
        if frame_pred_2_cut.shape[0] < max_pred_height:
            empty = np.full((max_pred_height, frame_pred_2_cut.shape[1], 3), 255, dtype=np.uint8)
            empty[-frame_pred_2_cut.shape[0]:] = frame_pred_2_cut
            frame_pred_2_cut = empty
        pred_cat = np.concatenate((frame_pred_0, frame_pred_1_cut, frame_pred_2_cut), axis=1)
        if height == orig_height:
            empty = np.full((height, max_pred_width + 2*(max_pred_width//2), 3), 255, dtype=np.uint8)
            empty[-pred_cat.shape[0]:, :] = pred_cat
            cat = np.concatenate((frame_orig, empty), axis=1)
        else:
            empty = np.full((height, orig_width, 3), 255, dtype=np.uint8)
            empty[-orig_height:, :] = frame_orig
            cat = np.concatenate((empty, pred_cat), axis=1)
        out_vid.write(cat)
        ret, frame_orig = cap_orig.read()
        ret1, frame_pred_0 = pred_caps[0].read()
        _, frame_pred_1 = pred_caps[1].read()
        _, frame_pred_2 = pred_caps[2].read()
        if not ret and not ret1:
            break
        elif not ret:
            frame_orig = np.ones((orig_height, orig_width, 3), dtype=np.uint8)
        elif not ret1:
            frame_pred_0 = np.ones(keypoints_shape, dtype=np.uint8)
            frame_pred_1 = np.ones(keypoints_shape, dtype=np.uint8)
            frame_pred_2 = np.ones(keypoints_shape, dtype=np.uint8)
        if scale and ret:
            frame_orig = cv2.resize(frame_orig, (orig_width, orig_height))
        i += 1
    out_vid.release()


def __get_features(input_image):
    from torchvision import models
    model = models.googlenet(pretrained=True)
    model.classifier = torch.nn.Sequential(*list(model.children())[:-1])
    model.eval()

    input_tensor = torch.from_numpy((input_image/255).transpose((0, 3, 1, 2))).float()

    # move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        input_tensor = input_tensor.to('cuda')
        model.to('cuda')

    with torch.no_grad():
        features = model.classifier(input_tensor)

    return features.squeeze()


def create_gaussian_pose_data(dataset, data_path, output_path):
    import pickle
    with open(data_path, 'r') as f:
        data_json = json.load(f)

    pickle_data = []
    for i, datum in enumerate(dataset):
        label_pose = datum["pose"]["obj"]
        frames = get_normalized_frames([label_pose], add_gaussians=True)[0]

        if datum['id'].isnumeric():  # only dgs for now
            features = __get_features(np.array(frames))
            pickle_data.append({
                "name": datum["id"],
                "signer": "dgs",
                "gloss": data_json[datum["id"]]["type_name"],
                "hamnosys": datum["text"],
                "text": "",
                "sign": features
            })

    test = pickle_data[:400]
    dev = pickle_data[400:500]
    train = pickle_data[500:]

    with open(output_path + ".train", 'wb') as of:
        pickle.dump(train, of)
    with open(output_path + ".dev", 'wb') as of:
        pickle.dump(dev, of)
    with open(output_path + ".test", 'wb') as of:
        pickle.dump(test, of)


if __name__ == '__main__':
    from torch.optim import Adam
    from torch.utils.data import DataLoader
    from text_to_pose.model import IterativeTextGuidedPoseGenerationModel
    from shared.collator import zero_pad_collator
    from text_to_pose.tokenizers.hamnosys.hamnosys_tokenizer import HamNoSysTokenizer

    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    DATASET_SIZE = 5750
    test_size = int(0.1 * DATASET_SIZE)
    print("test size", test_size)
    train_split = "train[:3]"  # f'test[{test_size}:]+train'
    # test_split = "train[:6]"  # f'test[:{test_size}]'

    dataset = get_dataset(name=args.dataset, poses=args.pose, fps=args.fps,
                          components=args.pose_components, exclude=True, use_relative_pose=False,
                          max_seq_size=args.max_seq_size, split=train_split)

    experiment_name = "reproduce_exclude_sep_2"#"exclude_bad_videos_sep_pos_embedding"
    _, num_pose_joints, num_pose_dims = dataset[0]["pose"]["data"].shape

    model_args = dict(tokenizer=HamNoSysTokenizer(),
                      pose_dims=(num_pose_joints, num_pose_dims),
                      hidden_dim=128,  # 256,#args.hidden_dim,
                      text_encoder_depth=args.text_encoder_depth,
                      pose_encoder_depth=args.pose_encoder_depth,
                      encoder_heads=args.encoder_heads,  # 4,
                      # encoder_dim_feedforward=512,
                      max_seq_size=args.max_seq_size,
                      num_steps=args.num_steps,
                      tf_p=args.tf_p,
                      masked_loss=args.masked_loss,
                      separate_positional_embedding=args.separate_positional_embedding,
                      num_pose_projection_layers=args.num_pose_projection_layers,
                      do_pose_self_attention=False,  # True,
                      use_transformer_decoder=False,  # True,
                      concat=True)

    args.checkpoint = f"/home/nlp/rotemsh/transcription/models/{experiment_name}/model.ckpt"
    model = IterativeTextGuidedPoseGenerationModel.load_from_checkpoint(args.checkpoint, **model_args)

    output_dir = f"/home/nlp/rotemsh/transcription/text_to_pose/videos/{experiment_name}/bsl"
    os.makedirs(output_dir, exist_ok=True)
    pose_header = dataset.data[0]["pose"].header

    bsl_id2hamnosys = {"cat": "", "nose": "", "accomodate": "",
                       "look": "", "one": "", "actor": "",
                       "address": "", "bed": "", "beverage": "",
                       "blue": "", "boat": "", "house": "",
                       "kitchen": "",
                       "love": ""}
    id2hamnosys = {"how": "\ue0e9\ue000\ue00c\ue029\ue03d\ue0d0\ue0e2\ue082\ue0aa\ue03f\ue0e3",
                   "you": "\ue002\ue0e6\ue002\ue010\ue031\ue03d\ue089\ue0c6",
                   "good": "\ue000\ue00c\ue031\ue028\ue03e\ue051\ue089\ue0c6\ue0cc"}
    model.eval()
    first_pose = dataset[2]["pose"]["data"][0]
    with torch.no_grad():
        for id, text in id2hamnosys.items():
            seq_iter = model.forward(text=text, first_pose=first_pose)
            for j in range(model.num_steps):
                seq = next(seq_iter)
            visualize_seq(seq, pose_header, output_dir, f"{id}", label_pose=None)
            first_pose = seq[-10]
    # pred(model, dataset, output_dir)
