import importlib
from typing import List, Union, Dict
import numpy as np
import numpy.ma as ma
from tqdm import tqdm

import tensorflow_datasets as tfds
from sign_language_datasets.datasets import SignDatasetConfig

from pose_format import Pose
from pose_format.pose_header import PoseHeader
from pose_format.numpy.pose_body import NumPyPoseBody
from pose_format.utils.reader import BufferReader
from shared.pose_utils import pose_normalization_info, pose_hide_legs, pose_hide_low_conf
import json
import os

PJM_FRAME_WIDTH = 1280
with open("/home/nlp/rotemsh/transcription/shared/pjm_left_videos.json", 'r') as f:
    PJM_LEFT_VIDEOS_LST = json.load(f)


class ProcessedPoseDatum(Dict):
    id: str
    pose: Union[Pose, Dict[str, Pose]]
    tf_datum: dict


def get_tfds_dataset(name, poses="holistic", fps=25, split="train",
                     components: List[str] = None, data_dir=None, version="1.0.0", no_flip=False):
    if name == "hamnosys":
        dataset_module = importlib.import_module("datasets.hamnosys")
    else:
        dataset_module = importlib.import_module("sign_language_datasets.datasets." + name + "." + name)

    # Loading a dataset with custom configuration
    config = SignDatasetConfig(name=poses + "-" + str(fps),
                               version=version,  # Specific version
                               include_video=False,  # Download and load dataset videos
                               fps=fps,  # Load videos at constant fps
                               include_pose=poses)  # Download and load Holistic pose estimation
    tfds_dataset = tfds.load(name=name, builder_kwargs=dict(config=config), split=split, data_dir=data_dir)

    # pylint: disable=protected-access
    with open(dataset_module._POSE_HEADERS[poses], "rb") as buffer:
        pose_header = PoseHeader.read(BufferReader(buffer.read()))

    normalization_info = pose_normalization_info(pose_header)
    return [process_datum(datum, pose_header, normalization_info, no_flip, components) for datum in tqdm(tfds_dataset)]


def process_datum(datum, pose_header: PoseHeader, normalization_info, no_flip,
                  components: List[str] = None) -> ProcessedPoseDatum:
    tf_poses = {"": datum["pose"]} if "pose" in datum else datum["poses"]
    poses = {}
    for key, tf_pose in tf_poses.items():

        fps = int(datum["fps"].numpy())
        pose_body = NumPyPoseBody(fps, tf_pose["data"].numpy(), tf_pose["conf"].numpy())
        pose = Pose(pose_header, pose_body)
        if not no_flip and datum["id"] in PJM_LEFT_VIDEOS_LST:
            pose.body.data[..., 0] = (PJM_FRAME_WIDTH - pose.body.data[..., 0]) % PJM_FRAME_WIDTH

        # Get subset of components if needed
        if components and len(components) != len(pose_header.components):
            pose = pose.get_components(components)

        pose = pose.normalize(normalization_info)
        pose_hide_legs(pose)
        pose_hide_low_conf(pose)
        poses[key] = pose
    if "pose" in datum:
        datum["pose"]["fps"] = datum["fps"]
    return {
        "id": datum["id"].numpy().decode('utf-8'),
        "pose": poses[""] if "pose" in datum else poses,
        "tf_datum": datum
    }
