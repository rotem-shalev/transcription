import tensorflow as tf
import tensorflow_datasets as tfds
from sign_language_datasets.datasets.config import SignDatasetConfig
import numpy as np
import os
import re
import random
import pickle
import importlib
from pose_format.pose_header import PoseHeader
from pose_format.utils.reader import BufferReader

import sys
sys.path.insert(0, '/home/nlp/rotemsh/transcription')

from shared.tfds_dataset import process_datum as process_datum_tfds
from text_to_pose.metrics import compare_poses, masked_mse, mse, APE
from text_to_pose.data import process_datum as process_datum_data
from text_to_pose.pred import visualize_pose
import matplotlib.pyplot as plt


def convert_to_137_format(d):
    body_135 = dict(enumerate(["Nose", "LEye", "REye", "LEar", "REar", "LShoulder", "RShoulder", "LElbow", "RElbow",
                         "LWrist", "RWrist", "LHip", "RHip", "LKnee", "RKnee", "LAnkle", "RAnkle", "Neck",
                          "MidHip", "LBigToe", "LSmallToe", "LHeel", "RBigToe", "RSmallToe", "RHeel"]))  # TODO-
    # switched "HeadTop" to "MidHip" because it's missing from the 135 pose
    body_137_lst = ["Nose", "Neck", "RShoulder", "RElbow", "RWrist", "LShoulder", "LElbow", "LWrist",
                              "MidHip", "RHip", "RKnee", "RAnkle", "LHip", "LKnee", "LAnkle", "REye", "LEye", "REar",
                               "LEar", "LBigToe", "LSmallToe", "LHeel", "RBigToe", "RSmallToe", "RHeel"]
    body_137 = {body_137_lst[i]: i for i in range(len(body_137_lst))}

    Lwrist = tf.expand_dims(d["pose"]["data"][:, :, 9], 2)
    Rwrist = tf.expand_dims(d["pose"]["data"][:, :, 10], 2)
    Lwrist_conf = tf.expand_dims(d["pose"]["conf"][:, :, 9], 2)
    Rwrist_conf = tf.expand_dims(d["pose"]["conf"][:, :, 10], 2)

    new_pose_data = np.zeros_like(d["pose"]["data"])
    new_pose_conf = np.zeros_like(d["pose"]["conf"])
    for idx, body_part in body_135.items():
        new_pose_data[:, :, body_137[body_part]] = d["pose"]["data"][:, :, idx]
        new_pose_conf[:, :, body_137[body_part]] = d["pose"]["conf"][:, :, idx]

    d["pose"]["data"] = tf.concat([new_pose_data[:, :, :25], Lwrist, new_pose_data[:, :, 25:45], Rwrist,
                                   new_pose_data[:, :, 45:]], axis=2)

    d["pose"]["conf"] = tf.concat([new_pose_conf[:, :, :25], Lwrist_conf, new_pose_conf[:, :, 25:45], Rwrist_conf,
                                   new_pose_conf[:, :, 45:]], axis=2)
    d["fps"] = d["pose"]["fps"]
    return d


def results_stats(path, limit=2000):
    with open(path, 'r') as f:
        results = f.read()

    precision_1_res = np.array([r[1] for r in re.findall("(precision@1: )(.*)", results)][:limit])
    precision_1 = sum(precision_1_res == "True")/len(precision_1_res)
    precision_5_res = np.array([float(r[1]) for r in re.findall("(precision@5: )(.*)", results)][:limit])
    precision_5_mean = precision_5_res.mean()
    precision_5_median = np.median(precision_5_res)
    precision_10_res = np.array([float(r[1]) for r in re.findall("(precision@10: )(.*)", results)][:limit])
    precision_10_mean = precision_10_res.mean()
    precision_10_median = np.median(precision_10_res)
    R_precision_res = np.array([float(r[1]) for r in re.findall("(R-precision: )(.*)", results)][:limit])
    R_precision_mean = R_precision_res.mean()
    R_precision_median = np.median(R_precision_res)
    mAP_res = np.array([float(r[1]) for r in re.findall("(mAP of .* is: )(.*)", results)][:limit])
    mAP_mean = mAP_res.mean()
    mAP_median = np.median(mAP_res)

    print(f"\nAUTSL eval statistics over {len(precision_1_res)} examples:\n")
    print(f"precision@1: {precision_1}")
    print(f"precision@5 mean: {precision_5_mean}, precision@5 median: {precision_5_median}")
    print(f"precision@10 mean: {precision_10_mean}, precision@10 median: {precision_10_median}")
    print(f"R-precision mean: {R_precision_mean}, R-precision median: {R_precision_median}")
    print(f"mAP mean: {mAP_mean}, mAP median: {mAP_median}")


if __name__ == "__main__":
    d = {"NMSE": "log_autsl_results_APE_new.txt", "MSE": "log_autsl_MSE.txt",
         # "normalized DTW": "log_autsl_results_DTW_new.txt",
         # "normalized WA-DTW": "log_autsl_results_WA-DTW_new.txt",
         # "WA-NDTW 2": "log_autsl_results_new_WA-NDTW_2.txt", "WA-NDTW": "log_autsl_results_new.txt", "NDTW": "log_autsl_results_NDTW_new.txt",
         # "WA-MSE": "log_autsl_results_WA-APE.txt", "WA-APE": "log_autsl_results_WA-APE_l2.txt"
         "normalized APE": "log_autsl_results_APE_l2.txt",
         # "DTW unnormalized": "log_autsl_results_WA-DTW_unnormalized.txt", "WA_DTW unnormalized":
         #     "log_autsl_results_WA_DTW_unnormalized.txt",
         "APE unnormalized": "log_autsl_results_APE_unnormalized.txt",
         # "WA_NDTW punish": "log_autsl_results_WA_DTW_punish_missing.txt",
         # "WA_NDTW punish half": "log_autsl_results_WA_DTW_punish_missing_half_norm.txt",
         # "DTW punish unnormalized": "log_autsl_results_DTW_punish_missing_unnorm.txt",
         "DTW punish half unnormalized": "log_autsl_results_DTW_punish_missing_half_unnorm.txt",
         "NDTW punish half": "log_autsl_results_DTW_punish_missing_half_norm.txt"}

    for experiment, filename in d.items():
        print("\n", experiment)
        results_stats(os.path.join("/home/nlp/rotemsh/transcription", filename), limit=500)
    # results_stats("/home/nlp/rotemsh/transcription/log_autsl_results_random.txt")
    exit()

    # for split in autsl_dataset:
    all_scores = []
    rank1 = []
    rank5 = []
    rank10 = []
    R_precision = []

    if True:
        split = "train"
        use_preprocessed = True
        if use_preprocessed and os.path.isfile("/home/nlp/rotemsh/transcription/processed_data_unnormalized.pkl"):
            with open("/home/nlp/rotemsh/transcription/processed_data_unnormalized.pkl", 'rb') as f:
                processed_data = pickle.load(f)
        else:
            config = SignDatasetConfig(name="only-annotations", version="1.0.0", include_video=False,
                                       include_pose="openpose")
            autsl_dataset = tfds.load(name='autsl', builder_kwargs={"config": config})

            dataset_module = importlib.import_module(f"sign_language_datasets.datasets.autsl.autsl")
            # pylint: disable=protected-access
            with open(dataset_module._POSE_HEADERS["openpose"], "rb") as buffer:
                pose_header = PoseHeader.read(BufferReader(buffer.read()))

            #####################
            # output_dir = f"/home/nlp/rotemsh/transcription/text_to_pose/videos/AUTSL"
            # for datum in autsl_dataset[split]:
            #     processed_data = process_datum_data(process_datum_tfds(convert_to_137_format(datum), pose_header))
            #     visualize_pose([processed_data["pose"]], f"{processed_data['text']}_{processed_data['id']}.mp4", output_dir)
            # exit()
            #####################

            processed_data = [convert_to_137_format(datum) for datum in autsl_dataset[split]]
            print("after 137")
            processed_data = [process_datum_tfds(datum, pose_header, normalize=False) for datum in processed_data]
            print("after first process")
            processed_data = [process_datum_data(d) for d in processed_data]
            print("after second process")

            with open("/home/nlp/rotemsh/transcription/processed_data_unnormalized.pkl", "wb") as f:
                pickle.dump(processed_data, f)

        print("after data processing")
        print("num signs: ", len(processed_data))
        for d1 in processed_data:
            signer = d1["id"].split("_")[0]
            if signer in ["signer20", "signer22", "signer8"]:
                continue
            print(f"\ndistances to {d1['id']}, gloss: {d1['text']}:")
            distances = dict()
            sign = d1["text"]

            test_same_signer = False
            if test_same_signer:
                same_sign = [d for d in processed_data if d["id"] != d1["id"] and d["id"].split("_")[0] == signer and
                             d["text"] == sign]
                same_sign_ids = [d["id"] + "_" + d["text"] for d in same_sign]
                other_signs = random.sample([d for d in processed_data if d["id"] != d1["id"] and d["id"].split("_")[0] == signer and
                             d["text"] != sign], 100)
            else:
                same_sign = [d for d in processed_data if d["text"] == sign and d["id"] != d1["id"] and
                                d["id"].split("_")[0] not in ["signer20", "signer22", "signer8"]]
                same_sign_ids = [d["id"]+"_"+d["text"] for d in same_sign]
                num_samples = len(same_sign_ids) * 4
                other_signs = random.sample([d for d in processed_data if (d["text"] != sign and
                                             d["id"].split("_")[0] not in ["signer20", "signer22", "signer8", signer])],
                                            num_samples)
            all_signs = same_sign + other_signs

            test_random = False
            if test_random:
                # random test
                random.shuffle(all_signs)
                rand_ranks = [i for i in range(len(all_signs)) if all_signs[i]["text"] == sign]
                print("precision@1:", rand_ranks[0] == 0)
                rank1.append(int(rand_ranks[0] == 0))
                cur_rank_5 = len([rank for rank in rand_ranks if rank < 5]) / 5
                rank5.append(cur_rank_5)
                print("precision@5:", cur_rank_5)
                cur_rank_10 = len([rank for rank in rand_ranks if rank < 10]) / 10
                rank10.append(cur_rank_10)
                print("precision@10:", cur_rank_10)

                cur_R_precision = len([rank for rank in rand_ranks if rank < len(same_sign_ids)]) / len(same_sign_ids)
                R_precision.append(cur_R_precision)
                print("R-precision:", cur_R_precision)
                rand_ranks_count = 0
                for i, rank in enumerate(rand_ranks):
                    rand_ranks_count += (i + 1) / (rank + 1)
                print("random mAP:", rand_ranks_count/len(rand_ranks))
                all_scores.append(rand_ranks_count/len(rand_ranks))
            else:
                for d2 in all_signs:
                    distance = compare_poses(d2["pose"], d1["pose"], distance_function=mse,
                                             normalize=False, more_hand_weight=False)
                    distances[d2["id"]+"_"+d2["text"]] = distance

                distances_lst = list(sorted(distances.items(), key=lambda item: item[1]))
                print(distances_lst)
                ranks = [i for i in range(len(distances_lst)) if distances_lst[i][0] in same_sign_ids]
                if not test_same_signer:
                    print("precision@1:", ranks[0] == 0)
                    rank1.append(int(ranks[0] == 0))
                    cur_rank_5 = len([rank for rank in ranks if rank < 5])/5
                    rank5.append(cur_rank_5)
                    print("precision@5:", cur_rank_5)
                    cur_rank_10 = len([rank for rank in ranks if rank < 10]) / 10
                    rank10.append(cur_rank_10)
                    print("precision@10:", cur_rank_10)

                cur_R_precision = len([rank for rank in ranks if rank < len(same_sign_ids)]) / len(same_sign_ids)
                R_precision.append(cur_R_precision)
                print("R-precision:", cur_R_precision)

                ranks_count = 0
                for i, rank in enumerate(ranks):
                    ranks_count += (i+1)/(rank+1)
                print(f"mAP of {d1['id']}, {d1['text']} is: {ranks_count/len(ranks)}")
                all_scores.append(ranks_count/len(ranks))

        print("\n")
        print("mean mAP: ", np.mean(all_scores))
        print("median mAP: ", np.median(all_scores))
        print("mean rank1: ", np.mean(rank1))
        print("median rank1: ", np.median(rank1))
        print("mean rank5: ", np.mean(rank5))
        print("median rank5: ", np.median(rank5))
        print("mean rank10: ", np.mean(rank10))
        print("median rank10: ", np.median(rank10))
        print("mean R-precision: ", np.mean(R_precision))
        print("median R-precision: ", np.median(R_precision))
