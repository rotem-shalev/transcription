import numpy as np
from numpy import ma
from pose_format import Pose
from pose_format.numpy import NumPyPoseBody
from pose_format.pose_header import PoseHeaderDimensions, PoseHeader
from pose_format.utils.openpose import OpenPose_Components


def pose_hide_low_conf(pose: Pose):
    mask = pose.body.confidence <= 0.2
    pose.body.confidence[mask] = 0
    stacked_confidence = np.stack([mask, mask], axis=3)
    masked_data = ma.masked_array(pose.body.data, mask=stacked_confidence)
    pose.body.data = masked_data


def pose_hide_legs(pose: Pose):
    if pose.header.components[0].name == "pose_keypoints_2d":
        point_names = ["Knee", "Ankle", "Heel", "BigToe", "SmallToe"]
        points = [pose.header._get_point_index(pose.header.components[0].name, side+n)
                for n in point_names for side in ["L", "R"]]
    elif pose.header.components[0].name == "POSE_LANDMARKS":
        point_names = ["KNEE", "ANKLE", "HEEL", "FOOT_INDEX"]
        # pylint: disable=protected-access
        points = [pose.header._get_point_index("POSE_LANDMARKS", side + "_" + n)
                  for n in point_names for side in ["LEFT", "RIGHT"]]
    else:
        raise ValueError("Unknown pose header schema for hiding legs")

    pose.body.confidence[:, :, points] = 0
    pose.body.data[:, :, points, :] = 0


def pose_normalization_info(pose_header: PoseHeader):
    if pose_header.components[0].name == "POSE_LANDMARKS":
        return pose_header.normalization_info(
            p1=("POSE_LANDMARKS", "RIGHT_SHOULDER"),
            p2=("POSE_LANDMARKS", "LEFT_SHOULDER")
        )

    if pose_header.components[0].name == "BODY_135":
        return pose_header.normalization_info(
            p1=("BODY_135", "RShoulder"),
            p2=("BODY_135", "LShoulder")
        )

    if pose_header.components[0].name == "pose_keypoints_2d":
        return pose_header.normalization_info(
            p1=("pose_keypoints_2d", "RShoulder"),
            p2=("pose_keypoints_2d", "LShoulder")
        )

    raise ValueError("Unknown pose header schema for normalization")


def fake_pose(num_frames: int, fps: int = 25):
    dimensions = PoseHeaderDimensions(width=1, height=1, depth=1)
    header = PoseHeader(version=0.1, dimensions=dimensions, components=OpenPose_Components)

    total_points = header.total_points()
    data = np.zeros(shape=(num_frames, 1, total_points, 2), dtype=np.float32)
    confidence = np.zeros(shape=(num_frames, 1, total_points), dtype=np.float32)
    masked_data = ma.masked_array(data)

    body = NumPyPoseBody(fps=int(fps), data=masked_data, confidence=confidence)

    return Pose(header, body)
