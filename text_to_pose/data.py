from typing import List, Dict

import torch
from pose_format import Pose
from torch.utils.data import Dataset
from shared.tfds_dataset import ProcessedPoseDatum, get_tfds_dataset


class TextPoseDatum(Dict):
    id: str
    text: str
    pose: Pose
    length: int


class TextPoseDataset(Dataset):
    def __init__(self, data: List[TextPoseDatum]):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        datum = self.data[index]
        if "pose" not in datum:
            return datum

        pose = datum["pose"]
        torch_body = pose.body.torch()
        pose_length = len(torch_body.data)

        return {
            "id": datum["id"],
            "text": datum["text"],
            "pose": {
                "obj": pose,
                "data": torch_body.data.tensor[:, 0, :, :],
                "confidence": torch_body.confidence[:, 0, :],
                "length": torch.tensor([pose_length], dtype=torch.float),
                "inverse_mask": torch.ones(pose_length, dtype=torch.bool)
            }
        }


def process_datum(datum: ProcessedPoseDatum) -> TextPoseDatum:
    text = datum["tf_datum"]["hamnosys"].numpy().decode('utf-8').strip() \
        if "hamnosys" in datum["tf_datum"] else ""
    text = str(datum["tf_datum"]["gloss_id"].numpy()) if "gloss_id" in datum["tf_datum"] else text
    if "pose" not in datum:
        return TextPoseDatum({
            "id": datum["id"],
            "text": text,
            "pose_len": float(datum["tf_datum"]["pose_len"].numpy()),
            "length": max(datum["tf_datum"]["pose_len"], len(text))
        })

    pose: Pose = datum["pose"]

    # Prune all leading frames containing only zeros, almost no face, or no hands
    for i in range(len(pose.body.data)):
        if pose.body.confidence[i][:, 25:-42].sum() >= 35 and \
                pose.body.confidence[i][:, 4] + pose.body.confidence[i][:, 7] >= 10:
            if i != 0:
                pose.body.data = pose.body.data[i:]
                pose.body.confidence = pose.body.confidence[i:]
            break

    # Prune all trailing frames containing only zeros, almost no face, or no hands
    for i in range(len(pose.body.data) - 1, 0, -1):
        if pose.body.confidence[i][:, 25:-42].sum() >= 35 and \
                pose.body.confidence[i][:, 4] + pose.body.confidence[i][:, 7] >= 10:
            if i != len(pose.body.data) - 1:
                pose.body.data = pose.body.data[:i + 1]
                pose.body.confidence = pose.body.confidence[:i + 1]
            break

    return TextPoseDatum({
        "id": datum["id"],
        "text": text,
        "pose": pose,
        "length": len(pose.body.data)
    })


def get_dataset(name="dicta_sign", poses="holistic", fps=25, split="train", include_low_conf_vids=True,
                use_relative_pose=False, exclude=False, components: List[str] = None, data_dir=None,
                max_seq_size=200, no_flip=False, leave_out=""):

    data = get_tfds_dataset(name=name, poses=poses, fps=fps, split=split, components=components,
                            exclude=exclude, data_dir=data_dir, no_flip=no_flip,
                            include_low_conf_vids=include_low_conf_vids, use_relative_pose=use_relative_pose)

    data = [process_datum(d) for d in data]
    data = [d for d in data if d["length"] < max_seq_size]
    if leave_out != "":
        if leave_out == "dgs":
            train_data = [d for d in data if not d["id"].isnumeric()]
            test_data = [d for d in data if d["id"].isnumeric()]
        elif leave_out == "lsf":
            train_data = [d for d in data if any(i.isdigit() for i in d["id"])]
            test_data = [d for d in data if not any(i.isdigit() for i in d["id"])]
        else:
            train_data = [d for d in data if leave_out not in d["id"]]
            test_data = [d for d in data if leave_out in d["id"]]
        return TextPoseDataset(train_data), TextPoseDataset(test_data)
    return TextPoseDataset(data)

#################################
# Augmentations
#################################

#
# class RepeatSign:
#
#     def __init__(self, p=0.5):
#         self.p = p
#
#     def __call__(self, sample):
#         REPEAT_FROM_START_CHAR = "\ue0d8"
#         if REPEAT_FROM_START_CHAR in sample["text"] or torch.rand(1) < self.p:
#             return sample
#         new_text = sample["text"] + REPEAT_FROM_START_CHAR
#         new_pose = sample["pose"]  # TODO
#         new_sample = {
#             "id": sample["id"],
#             "text": new_text,
#             "pose": new_pose
#         }
#         return new_sample
#
#
# if __name__ == "__main__":
#     import torchvision
#     transform = torchvision.transforms.Compose([
#         RepeatSign(p=1),
#         torchvision.transforms.ToTensor(),
#     ])
#     test_dataset = get_dataset(name="hamnosys", poses="openpose", fps=25,
#                                     components=None,
#                                     max_seq_size=100, split="train[10:20]")
#     transform(test_dataset[0])
