"""hamnosys dataset."""

import tensorflow_datasets as tfds
import tensorflow as tf
from os import path
import json
import numpy as np
import numpy.ma as ma

from typing import Dict
from pose_format.utils.openpose import load_openpose
from pose_format.pose import Pose

import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
sys.path.append(os.path.join(os.path.dirname(SCRIPT_DIR), ".."))
sys.path.append(os.path.join(os.path.dirname(SCRIPT_DIR), "..", "utils"))

from ..config import SignDatasetConfig
from ..utils.features import PoseFeature


_DESCRIPTION = """
Combined corpus of videos with their openPose keypoints and HamNoSys. 
Includes dicta_sign, KSPJM (The Corpus Dictionary of Polish Sign Language).
"""

_CITATION = """
@inproceedings{efthimiou2010dicta,
  title={Dicta-sign--sign language recognition, generation and modelling: a research effort with applications in deaf communication},
  author={Efthimiou, Eleni and Fontinea, Stavroula-Evita and Hanke, Thomas and Glauert, John and Bowden, Rihard and Braffort, Annelies and Collet, Christophe and Maragos, Petros and Goudenove, Fran{\c{c}}ois},
  booktitle={Proceedings of the 4th Workshop on the Representation and Processing of Sign Languages: Corpora and Sign Language Technologies},
  pages={80--83},
  year={2010}
}

@inproceedings{linde2014corpus,
  title={A corpus-based dictionary of Polish Sign Language (PJM)},
  author={Linde-Usiekniewicz, Jadwiga and Czajkowska-Kisil, Ma{\l}gorzata and {\L}acheta, Joanna and Rutkowski, Pawe{\l}},
  booktitle={Proceedings of the XVI EURALEX International Congress: The user in focus},
  pages={365--376},
  year={2014}
}
"""

MAX_HEIGHT = 400
MAX_WIDTH = 400

_POSE_HEADERS = {
    "openpose": path.join(path.dirname(path.realpath(__file__)), "openpose.poseheader"),
}
_DATA_PATH = path.join(path.dirname(path.realpath(__file__)), "data.json")
# _KEYPOINTS_PATH = path.join(path.dirname(path.realpath(__file__)), "keypoints")
_KEYPOINTS_PATH = "/home/nlp/rotemsh/SLP/data/keypoints_dir"


def get_pose(keypoints_path: str, fps: int) -> Dict[str, Pose]:
    """
    Load OpenPose in the particular format used by DGS (one single file vs. one file for each frame).

    :param keypoints_path: Path to a folder that contains keypoints jsons (OpenPose output)
    for all frames of a video.
    :param fps: frame rate.
    :return: Dictionary of Pose object with a header specific to OpenPose and a body that contains a
    single array.
    """

    files = sorted(tf.io.gfile.listdir(keypoints_path))
    frames = dict()
    try:
        for i, file in enumerate(files):
            with tf.io.gfile.GFile(path.join(keypoints_path, file), "r") as openpose_raw:
                frame_json = json.load(openpose_raw)
                frames[i] = {"people": frame_json["people"], "frame_id": i}
    except:
        print(keypoints_path)
        return None
    if frames == {}:
        print(keypoints_path)
        return None
    # Convert to pose format
    pose = load_openpose(frames, fps=fps, width=400, height=400)
    return pose


class Hamnosys(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for hamnosys dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
        '1.0.0': 'Initial release.',
    }

    BUILDER_CONFIGS = [
        SignDatasetConfig(name="default", include_video=False, include_pose="openpose"),
        SignDatasetConfig(name="videos", include_video=True, include_pose="openpose"),
    ]

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        pose_header_path = _POSE_HEADERS[self._builder_config.include_pose]
        stride = 1 if self._builder_config.fps is None else 50 / self._builder_config.fps
        pose_shape = (None, 1, 137, 2)

        features = tfds.features.FeaturesDict({
            "id": tfds.features.Text(),
            "video": tfds.features.Text(), #tfds.features.Video(shape=(None, MAX_HEIGHT, MAX_WIDTH, 3)),
            "fps": tf.int32,
            "pose": PoseFeature(shape=pose_shape, stride=stride, header_path=pose_header_path),
            "hamnosys": tfds.features.Text(),
            "text": tfds.features.Text()
        })

        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=features,
            # If there's a common (input, target) tuple from the
            # features, specify them here. They'll be used if
            # `as_supervised=True` in `builder.as_dataset`.
            supervised_keys=None,  # Set to `None` to disable
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        # TODO(hamnosys): Downloads the data and defines the splits
        # path = dl_manager.download_and_extract('https://todo-data-url')
        with tf.io.gfile.GFile(_DATA_PATH) as f:
            data = json.load(f)
        return [tfds.core.SplitGenerator(name=tfds.Split.TRAIN, gen_kwargs={"data": data})]

    def _generate_examples(self, data):
        """Yields examples."""

        default_fps = 25
        for key, val in data.items():
            if not tf.io.gfile.isdir(path.join(_KEYPOINTS_PATH, key)):
                continue
            features = {
                "id": key,
                "video": val["video_frontal"],
                "fps": default_fps,
                "pose": get_pose(path.join(_KEYPOINTS_PATH, key), fps=default_fps),
                "hamnosys": val["hamnosys"],
                "text": val["type_name"]
            }
            yield key, features  # TODO- should key be running id?
