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
    text = datum["tf_datum"]["hamnosys"].numpy().decode('utf-8').strip()
    pose: Pose = datum["pose"]

    # Prune all leading frames containing only zeros
    for i in range(len(pose.body.data)):
        if pose.body.confidence[i].sum() != 0:
            if i != 0:
                pose.body.data = pose.body.data[i:]
                pose.body.confidence = pose.body.confidence[i:]
            break

    return TextPoseDatum({
        "id": datum["id"],
        "text": text,
        "pose": pose,
        "length": max(len(pose.body.data), len(text))
        })


def get_dataset(name="dicta_sign", poses="holistic", fps=25, split="train",
                components: List[str] = None, data_dir=None, max_seq_size=1000, no_flip=False):
    data = get_tfds_dataset(name=name, poses=poses, fps=fps, split=split, components=components, data_dir=data_dir,
                            no_flip=no_flip)

    data = [process_datum(d) for d in data]
    data = [d for d in data if d["length"] < max_seq_size]

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
