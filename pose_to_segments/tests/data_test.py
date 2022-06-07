import unittest
from typing import List

import torch

from pose_to_segments.data import PoseSegmentsDatum, Segment, PoseSegmentsDataset
from shared.pose_utils import fake_pose


def single_datum(num_frames, segments: List[List[Segment]]) -> PoseSegmentsDatum:
    return {
        "id": "test_id",
        "pose": fake_pose(num_frames=num_frames),
        "segments": segments
    }


class DataTestCase(unittest.TestCase):

    def test_item_without_segments(self):
        datum = single_datum(num_frames=5, segments=[])
        dataset = PoseSegmentsDataset([datum])
        self.assertEqual(len(dataset), 1)

        pose = dataset[0]["pose"]
        self.assertEqual(pose["data"].shape, (5, 137, 2))

        for bio_type in ["sign", "sentence"]:
            bio = dataset[0][bio_type + "_bio"]
            self.assertEqual(bio.shape, tuple([5]))
            self.assertTrue(torch.all(torch.eq(torch.zeros_like(bio), bio)))


if __name__ == '__main__':
    unittest.main()
