"""hamnosys dataset."""

import tensorflow_datasets as tfds
import numpy as np
import tensorflow as tf
from os import path
import json

from typing import Dict
from pose_format.utils.openpose import load_openpose
from pose_format.pose import Pose

import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
sys.path.append(os.path.join(os.path.dirname(SCRIPT_DIR), ".."))
sys.path.append(os.path.join(os.path.dirname(SCRIPT_DIR), "..", "utils"))

from ..config import SignDatasetConfig  # for compilation remove ".."
from ..utils.features import PoseFeature  # for compilation remove ".."

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

EXCLUDE_VIDEOS = ["pjm_1969", "pjm_1066", "pjm_3052", "gsl_602", "pjm_2813", "pjm_1563", "pjm_255", "pjm_1099",
                  "pjm_143", "gsl_204", "pjm_1476", "pjm_3076", "pjm_3370", "pjm_2269", "ENTIER_ ", "pjm_1523",
                  "pjm_1199", "pjm_2737", "pjm_2901", "pjm_262", "pjm_444", "pjm_2895", "gsl_67", "pjm_1455",
                  "gsl_325", "pjm_2852", "gsl_644", "pjm_951", "gsl_842", "gsl_597", "pjm_2080", "pjm_1173",
                  "gsl_736", "gsl_163", "pjm_331", "gsl_76", "pjm_1213", "gsl_624", "gsl_243", "66239", "72062",
                  "44582", "pjm_1011", "gsl_558", "pjm_3320", "gsl_838", "gsl_646", "24945", "gsl_173", "pjm_1024",
                  "gsl_102", "pjm_740", "pjm_1641", "pjm_1543", "pjm_2670", "pjm_625", "pjm_501", "pjm_1247",
                  "gsl_665", "pjm_1409", "pjm_3409", "pjm_837", "pjm_571", "pjm_1701", "pjm_1177", "pjm_3040",
                  "pjm_3377", "gsl_321", "pjm_3131", "pjm_878", "pjm_1400", "pjm_3186", "pjm_1473", "pjm_1676",
                  "pjm_1739", "pjm_383", "pjm_3366", "pjm_1632", "pjm_2221", "gsl_196", "pjm_66", "pjm_3094",
                  "40181", "pjm_1257", "pjm_810", "pjm_436", "pjm_1607", "pjm_1144", "pjm_2952", "gsl_308",
                  "pjm_750", "pjm_3082", "gsl_74", "pjm_3018", "pjm_411", "pjm_2087", "gsl_787", "pjm_1104",
                  "pjm_1427", "pjm_1015", "pjm_1179", "gsl_845", "29531", "gsl_341", "gsl_730", "63674", "pjm_2306",
                  "pjm_726", "pjm_2781", "pjm_72", "pjm_868", "gsl_803", "pjm_2674", "gsl_677", "gsl_876", "gsl_35",
                  "pjm_3176", "pjm_2072", "pjm_538", "pjm_1017", "gsl_448", "pjm_1299", "gsl_758", "pjm_1894",
                  "pjm_733", "gsl_989", "pjm_201", "pjm_3059", "gsl_425", "pjm_1915", "pjm_2214", "gsl_313",
                  "gsl_1009", "pjm_3291", "pjm_420", "pjm_513", "gsl_1027", "gsl_289", "pjm_2280", "pjm_241",
                  "pjm_3003", "pjm_640", "pjm_235", "pjm_1083", "pjm_2897", "pjm_714", "pjm_1209", "pjm_3420",
                  "pjm_1759", "pjm_1992", "pjm_760", "gsl_884", "pjm_2162", "gsl_315", "gsl_521", "pjm_3110",
                  "pjm_1883", "gsl_8", "2671", "pjm_862", "pjm_1942", "gsl_23", "pjm_1708", "pjm_260", "56970",
                  "pjm_535", "pjm_1527", "gsl_37", "gsl_178", "pjm_1648", "pjm_985", "gsl_1022", "pjm_2801",
                  "pjm_793", "pjm_1754", "pjm_2813", "pjm_3069", "pjm_148", "pjm_1926", "pjm_2906", "pjm_1429",
                  "pjm_519", "gsl_777", "pjm_2395", "gsl_30", "pjm_3333", "pjm_1609", "pjm_1710", "pjm_616",
                  "gsl_295", "pjm_1258", "pjm_452", "pjm_1563", "gsl_767", "gsl_789", "pjm_2193", "gsl_848",
                  "gsl_974", "gsl_129", "gsl_620", "gsl_253", "pjm_1255", "pjm_1810", "pjm_703", "pjm_493",
                  "pjm_774", "pjm_2222", "pjm_1313", "gsl_432", "gsl_1023", "pjm_2521", "gsl_792", "pjm_3047",
                  "pjm_1605", "pjm_3476", "pjm_2331", "pjm_1113", "gsl_895", "pjm_391", "pjm_741", "pjm_3004",
                  "pjm_606", "pjm_3099", "gsl_633", "pjm_1346", "pjm_1914", "pjm_3183", "gsl_851", "pjm_3447",
                  "pjm_1811", "gsl_1019", "gsl_62", "pjm_979", "gsl_980", "gsl_629", "gsl_708", "gsl_781",
                  "pjm_1036", "pjm_3209", "gsl_142", "pjm_738", "pjm_1099", "gsl_1015", "gsl_293", "pjm_1001",
                  "gsl_343", "pjm_1342", "pjm_448", "pjm_208", "pjm_1440", "pjm_1389", "gsl_229", "pjm_3437",
                  "gsl_556", "pjm_3014", "gsl_177", "gsl_329", "pjm_330", "pjm_2851", "gsl_150", "pjm_1537",
                  "pjm_2213", "pjm_3313", "pjm_1718", "pjm_580", "gsl_10", "pjm_78", "pjm_2017", "pjm_3403",
                  "pjm_1180", "pjm_2074", "gsl_231", "pjm_3283", "pjm_1520", "pjm_2059", "pjm_2412", "pjm_2894",
                  "gsl_801", "pjm_891", "pjm_797", "pjm_1309", "pjm_2104", "pjm_1184", "gsl_94", "gsl_368",
                  "pjm_3240", "pjm_3127", "pjm_2124", "gsl_676", "pjm_169", "pjm_20", "gsl_909", "pjm_1757",
                  "gsl_710", "pjm_1062", "gsl_599", "pjm_3032", "pjm_2394", "gsl_300", "pjm_3351", "pjm_2174",
                  "pjm_2363", "gsl_119", "pjm_1059", "pjm_1528", "pjm_2296", "pjm_3434", "pjm_390", "pjm_756",
                  "pjm_889", "pjm_1431", "gsl_19", "pjm_3343", "pjm_438", "pjm_1043", "gsl_690", "pjm_2493",
                  "pjm_1319", "pjm_3186", "pjm_1911", "pjm_1473", "pjm_2461", "gsl_418", "gsl_204", "pjm_3116",
                  "pjm_1766", "pjm_1118", "pjm_1338", "pjm_789", "gsl_713", "pjm_609", "pjm_1962", "gsl_967",
                  "pjm_1988", "pjm_1226", "gsl_508", "pjm_2165", "pjm_1046", "gsl_856", "gsl_437", "pjm_2285",
                  "pjm_1421", "pjm_3138", "SUPPLIER", "pjm_498", "gsl_695", "pjm_1476", "pjm_1425", "pjm_1454",
                  "pjm_1324", "gsl_847", "pjm_1597", "pjm_355", "pjm_2827", "pjm_61", "pjm_902", "pjm_653",
                  "pjm_1238", "pjm_2399", "pjm_1661", "gsl_706", "pjm_2481", "pjm_2054", "gsl_729", "gsl_634",
                  "gsl_678", "pjm_570", "pjm_3136", "pjm_1475", "pjm_33", "pjm_32", "gsl_359", "pjm_2123",
                  "pjm_2642", "pjm_162", "pjm_3255", "pjm_2156", "pjm_81", "pjm_3011", "gsl_453", "pjm_346",
                  "pjm_1628", "pjm_298", "gsl_414", "pjm_3375", "pjm_1531", "gsl_658", "pjm_2140", "pjm_3459",
                  "46613", "pjm_749", "pjm_840", "pjm_812", "gsl_505", "pjm_1485", "pjm_3070", "pjm_131", "gsl_297",
                  "pjm_2013", "pjm_536", "pjm_3109", "TROMPER", "pjm_1011", "pjm_3226", "gsl_615", "pjm_2282",
                  "pjm_844", "gsl_114", "pjm_1069", "gsl_59", "pjm_353", "pjm_1455", "gsl_67", "pjm_1132",
                  "pjm_1422", "gsl_538", "pjm_871", "pjm_1612", "pjm_3305", "pjm_3319", "gsl_399", "pjm_1550",
                  "15424", "gsl_586", "pjm_1131", "pjm_2248", "pjm_3148", "pjm_1115", "pjm_1424", "pjm_588",
                  "pjm_21", "RADIO", "SE_TENIR_DEBOUT", "RIRE", "47681", "gsl_603", "APRE_MIDI", "28022", "58349",
                  "57873", "54400", "3779", "3497", "31934", "3192", "3075", "2830", "VISITE", "TROMPER_", "TACHE",
                  "SAUTER", "REPETER_", "10452", "13910", "14130", "19585", "27217", "5773", "SURPRISE", "60459",
                  "42859", "RADIATOUR", "MEME", "gsl_165", "2686"]

DUP_KEYS = ['10248', '3516', '44909', '10573', '12916', '2674', '10753', '8044', '10890', '69225', '9280', '11286', '48575', '68699', '11288', '27428', '6248', '11291', '75271', '11420', '39949', '11435', '59785', '6230', '11874', '2294', '12278', '3071', '12641', '59684', '12844', '59701', '15121', '85192', '15286', '59212', '15735', '20652', '15962', '2803', '16153', '40233', '17265', '67630', '18003', '89436', '2442', '3048', '9028', '2452', '2856', '25235', '4511', '2686', '5035', '27521', '87394', '29817', '86689', '30365', '4171', '3172', '40005', '5908', '3193', '88457', '43516', '65542', '48749', '68018', '53036', '9386', '5492', '91376', '55848', '72736', '56000', '76667', '56684', '58318', '59424', '6192', '60848', '73060', '61731', '7247', '8291', '71120', '85160', '76557', '80774', '7940', '9790', '8265', '87255', '8289', '87848', 'FEUILLE', 'PAPIER', 'gsl_1024', 'gsl_165', 'gsl_124', 'gsl_51', 'gsl_145', 'gsl_804', 'gsl_148', 'gsl_212', 'gsl_189', 'gsl_318', 'gsl_236', 'gsl_585', 'gsl_244', 'gsl_965', 'gsl_27', 'gsl_504', 'gsl_339', 'gsl_530', 'gsl_353', 'gsl_719', 'gsl_424', 'gsl_923', 'gsl_475', 'gsl_545', 'gsl_495', 'gsl_883', 'gsl_528', 'gsl_692']

_POSE_HEADERS = {
    "openpose": path.join(path.dirname(path.realpath(__file__)), "openpose.poseheader"),
}
_DATA_PATH = path.join(path.dirname(path.realpath(__file__)), "data.json")
_KEYPOINTS_PATH = path.join(path.dirname(path.realpath(__file__)), "keypoints")
# _KEYPOINTS_PATH = "/home/nlp/rotemsh/SLP/data/keypoints_dir"
MIN_CONFIDENCE = 0.2


def get_pose(keypoints_path: str, fps: int) -> Dict[str, Pose]:
    """
    Load OpenPose in the particular format used by DGS (one single file vs. one file for each frame).

    :param keypoints_path: Path to a folder that contains keypoints jsons (OpenPose output)
    for all frames of a video.
    :param fps: frame rate.
    :return: Dictionary of Pose object with a header specific to OpenPose and a body that contains a
    single array.
    """
    fps = 25
    files = sorted(tf.io.gfile.listdir(keypoints_path))
    frames = dict()
    for i, file in enumerate(files):
        try:
            with tf.io.gfile.GFile(path.join(keypoints_path, file), "r") as openpose_raw:
                frame_json = json.load(openpose_raw)
                frames[i] = {"people": frame_json["people"][:1], "frame_id": i}
                cur_frame_pose = frame_json["people"][0]
                if (np.array(cur_frame_pose['pose_keypoints_2d'][7*3:7*3 + 2]) -
                    np.array(cur_frame_pose['hand_left_keypoints_2d'][0:2])).max() > 15 and \
                        cur_frame_pose['pose_keypoints_2d'][7*3 + 2] > MIN_CONFIDENCE:
                    cur_frame_pose['hand_left_keypoints_2d'][0:2] = cur_frame_pose['pose_keypoints_2d'][7*3:7*3 + 2]
                if (np.array(cur_frame_pose['pose_keypoints_2d'][4*3:4*3 + 2]) -
                    np.array(cur_frame_pose['hand_right_keypoints_2d'][0:2])).max() > 15 and \
                        cur_frame_pose['pose_keypoints_2d'][4*3 + 2] > MIN_CONFIDENCE:
                    cur_frame_pose['hand_right_keypoints_2d'][0:2] = cur_frame_pose['pose_keypoints_2d'][4*3:4*3 + 2]

        except:
            continue
            # print(keypoints_path)
            # return None
    if len(frames) == 0:
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
        SignDatasetConfig(name="default", include_video=False, include_pose="openpose", fps=25),
        SignDatasetConfig(name="videos", include_video=True, include_pose="openpose"),
        SignDatasetConfig(name="text_only", include_video=False, include_pose=None, fps=0),
    ]

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        if self._builder_config.include_pose is None:
            features = tfds.features.FeaturesDict({
                "id": tfds.features.Text(),
                "hamnosys": tfds.features.Text(),
                "text": tfds.features.Text(),
                "pose_len": tf.float32,
            })
        else:
            pose_header_path = _POSE_HEADERS[self._builder_config.include_pose]
            stride = 1  # if self._builder_config.fps is None else 50 / self._builder_config.fps
            pose_shape = (None, 1, 137, 2)
            features = tfds.features.FeaturesDict({
                "id": tfds.features.Text(),
                "video": tfds.features.Text(),  # tfds.features.Video(shape=(None, MAX_HEIGHT, MAX_WIDTH, 3)),
                "fps": tf.int32,
                "pose": PoseFeature(shape=pose_shape, stride=stride, header_path=pose_header_path),
                "hamnosys": tfds.features.Text(),
                "text": tfds.features.Text(),
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
        # Downloads the data and defines the splits
        with tf.io.gfile.GFile(_DATA_PATH) as f:
            data = json.load(f)
        return [tfds.core.SplitGenerator(name=tfds.Split.TRAIN, gen_kwargs={"data": data, "is_train": True}),
                tfds.core.SplitGenerator(name=tfds.Split.TEST, gen_kwargs={"data": data, "is_train": False})]

    def _generate_examples(self, data, is_train):
        """Yields examples."""
        default_fps = 25
        i = 0
        for key, val in data.items():
            if not tf.io.gfile.isdir(path.join(_KEYPOINTS_PATH, key)):
                continue
            if "exclude" in self._builder_config.name and key in EXCLUDE_VIDEOS:
                continue
            if "high_conf_vids" in self._builder_config.name and not key.isnumeric():  # only include dgs vids
                continue
            if (is_train and key not in DUP_KEYS) or (not is_train and key in DUP_KEYS):
                continue

            features = {
                "id": key,
                "hamnosys": val["hamnosys"],
                "text": val["type_name"]
            }

            if self._builder_config.include_pose is not None:
                features["video"] = val["video_frontal"]
                features["fps"] = default_fps
                features["pose"] = get_pose(path.join(_KEYPOINTS_PATH, key), fps=default_fps)
            else:
                features["pose_len"] = len(tf.io.gfile.listdir(path.join(_KEYPOINTS_PATH, key)))

            i += 1
            yield key, features
