import importlib
from typing import List, Union, Dict
import numpy as np
from tqdm import tqdm

import tensorflow_datasets as tfds
from sign_language_datasets.datasets import SignDatasetConfig

from pose_format import Pose
from pose_format.pose_header import PoseHeader
from pose_format.numpy.pose_body import NumPyPoseBody
from pose_format.utils.reader import BufferReader
from shared.pose_utils import pose_normalization_info, pose_hide_legs, pose_hide_low_conf, get_relative_pose
import json

PJM_FRAME_WIDTH = 1280
with open("/home/nlp/rotemsh/transcription/shared/pjm_left_videos.json", 'r') as f:
    PJM_LEFT_VIDEOS_LST = json.load(f)


class ProcessedPoseDatum(Dict):
    id: str
    pose: Union[Pose, Dict[str, Pose]]
    tf_datum: dict


def get_tfds_dataset(name, poses="holistic", fps=25, split="train", include_low_conf_vids=True, exclude=False,
                     components: List[str] = None, data_dir=None, version="1.0.0", no_flip=False, use_relative_pose=False):
    if name in ["hamnosys", "perfect_pose_frames"]:
        dataset_module = importlib.import_module(f"datasets.{name}.{name}")
    else:
        dataset_module = importlib.import_module(f"sign_language_datasets.datasets.{name}.{name}")

    # Loading a dataset with custom configuration
    config_name = "text-only" if poses is None else poses + "-" + str(fps)
    config_name = config_name if include_low_conf_vids else config_name + "-high_conf_vids"
    config_name = config_name if not exclude else config_name + "-exclude"
    config = SignDatasetConfig(name=config_name,
                               version=version,  # Specific version
                               include_video=False,  # Download and load dataset videos
                               fps=fps,  # Load videos at constant fps
                               include_pose=poses)   # Download and load pose estimation
    tfds_dataset = tfds.load(name=name, builder_kwargs=dict(config=config), split=split, data_dir=data_dir)

    if poses is None:
        return [{"id": datum["id"].numpy().decode('utf-8'), "tf_datum": datum} for datum in tqdm(tfds_dataset)]

    # pylint: disable=protected-access
    with open(dataset_module._POSE_HEADERS[poses], "rb") as buffer:
        pose_header = PoseHeader.read(BufferReader(buffer.read()))

    return [process_datum(datum, pose_header, no_flip, components, use_relative_pose) for datum in tqdm(tfds_dataset)]


def swap_coords(pose, idx1, idx2):
    if type(idx1) == tuple:
        pose.body.data[:, :, idx1[0]:idx1[1]], pose.body.data[:, :, idx2[0]:idx2[1]] = \
        pose.body.data[:, :, idx2[0]:idx2[1]].copy(), pose.body.data[:, :, idx1[0]:idx1[1]].copy()
        pose.body.confidence[:, :, idx1[0]:idx1[1]], pose.body.confidence[:, :, idx2[0]:idx2[1]] = \
        pose.body.confidence[:, :, idx2[0]:idx2[1]].copy(), pose.body.confidence[:, :, idx1[0]:idx1[1]].copy()
    else:
        pose.body.data[:, :, idx1], pose.body.data[:, :, idx2] = \
            pose.body.data[:, :, idx2].copy(), pose.body.data[:, :, idx1].copy()
        pose.body.confidence[:, :, idx1], pose.body.confidence[:, :, idx2] = \
            pose.body.confidence[:, :, idx2].copy(), pose.body.confidence[:, :, idx1].copy()
    return pose


def flip_pose(pose):
    pose = pose.flip(axis=0)
    # body
    pose = swap_coords(pose, (2, 5), (5, 8))
    pose = swap_coords(pose, 9, 12)
    pose = swap_coords(pose, 15, 16)
    pose = swap_coords(pose, 17, 18)
    num_keypoints_body = 25
    # face
    for i in range(8):
        pose = swap_coords(pose, i + num_keypoints_body, 16 - i + num_keypoints_body)
    for i in range(17, 22):
        pose = swap_coords(pose, i + num_keypoints_body, 26 - i + 17 + num_keypoints_body)
    # eyes
    pose = swap_coords(pose, 36 + num_keypoints_body, 45 + num_keypoints_body)
    pose = swap_coords(pose, 41 + num_keypoints_body, 46 + num_keypoints_body)
    pose = swap_coords(pose, 40 + num_keypoints_body, 47 + num_keypoints_body)
    pose = swap_coords(pose, 37 + num_keypoints_body, 44 + num_keypoints_body)
    pose = swap_coords(pose, 38 + num_keypoints_body, 43 + num_keypoints_body)
    pose = swap_coords(pose, 39 + num_keypoints_body, 42 + num_keypoints_body)
    pose = swap_coords(pose, 68 + num_keypoints_body, 69 + num_keypoints_body)
    # nose
    pose = swap_coords(pose, 31 + num_keypoints_body, 35 + num_keypoints_body)
    pose = swap_coords(pose, 32 + num_keypoints_body, 34 + num_keypoints_body)
    # mouth
    pose = swap_coords(pose, 50 + num_keypoints_body, 52 + num_keypoints_body)
    pose = swap_coords(pose, 49 + num_keypoints_body, 53 + num_keypoints_body)
    pose = swap_coords(pose, 48 + num_keypoints_body, 54 + num_keypoints_body)
    pose = swap_coords(pose, 59 + num_keypoints_body, 55 + num_keypoints_body)
    pose = swap_coords(pose, 58 + num_keypoints_body, 56 + num_keypoints_body)
    pose = swap_coords(pose, 61 + num_keypoints_body, 63 + num_keypoints_body)
    pose = swap_coords(pose, 60 + num_keypoints_body, 64 + num_keypoints_body)
    pose = swap_coords(pose, 67 + num_keypoints_body, 65 + num_keypoints_body)
    # hands
    pose = swap_coords(pose, (-21, pose.body.data.shape[2]), (-42, -21))
    pose.body.data = pose.body.data.astype(np.float32)
    return pose


def process_datum(datum, pose_header: PoseHeader, no_flip, components: List[str] = None,
                  use_relative_pose: bool = False) -> ProcessedPoseDatum:
    tf_poses = {"": datum["pose"]} if "pose" in datum else datum["poses"]
    poses = {}
    for key, tf_pose in tf_poses.items():
        fps = int(datum["fps"].numpy())
        pose_body = NumPyPoseBody(fps, tf_pose["data"].numpy(), tf_pose["conf"].numpy())
        pose = Pose(pose_header, pose_body)

        if not no_flip and datum["id"] in PJM_LEFT_VIDEOS_LST:
            pose = flip_pose(pose)

        # Get subset of components if needed
        if components and len(components) != len(pose_header.components):
            pose = pose.get_components(components)

        normalization_info = pose_normalization_info(pose_header)
        pose = pose.normalize(normalization_info)
        if not components or "pose_keypoints_2d" in components:
            pose_hide_legs(pose)
        pose_hide_low_conf(pose)

        if use_relative_pose:
            pose.body.data = get_relative_pose(pose.body.data)

        poses[key] = pose
    if "pose" in datum:
        datum["pose"]["fps"] = datum["fps"]
    return {
        "id": datum["id"].numpy().decode('utf-8'),
        "pose": poses[""] if "pose" in datum else poses,
        "tf_datum": datum
    }
