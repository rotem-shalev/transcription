import os
import json
import pickle
import random
import numpy as np
from numpy import cov, trace, iscomplexobj
from scipy.linalg import sqrtm
import importlib
import matplotlib.pyplot as plt
import tensorflow as tf
import torch
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw

from pose_format.utils.openpose import load_openpose
from pose_format import Pose
from pose_format.pose_header import PoseHeader
from pose_format.numpy.pose_body import NumPyPoseBody
from pose_format.utils.reader import BufferReader
from shared.pose_utils import pose_normalization_info, pose_hide_legs, pose_hide_low_conf, get_relative_pose
from shared.tfds_dataset import flip_pose

PJM_FRAME_WIDTH = 1280
with open("/home/nlp/rotemsh/transcription/shared/pjm_left_videos.json", 'r') as f:
    PJM_LEFT_VIDEOS_LST = json.load(f)

dataset_module = importlib.import_module(f"datasets.hamnosys.hamnosys")

with open(dataset_module._POSE_HEADERS["openpose"], "rb") as buffer:
    pose_header = PoseHeader.read(BufferReader(buffer.read()))


# utils
def get_pose(keypoints_path: str, datum_id):
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
            with tf.io.gfile.GFile(os.path.join(keypoints_path, file), "r") as openpose_raw:
                frame_json = json.load(openpose_raw)
                frames[i] = {"people": frame_json["people"][:1], "frame_id": i}
                cur_frame_pose = frame_json["people"][0]
                if (np.array(cur_frame_pose['pose_keypoints_2d'][7*3:7*3 + 2]) -
                    np.array(cur_frame_pose['hand_left_keypoints_2d'][0:2])).max() > 15 and \
                        cur_frame_pose['pose_keypoints_2d'][7*3 + 2] > 0.2:
                    cur_frame_pose['hand_left_keypoints_2d'][0:2] = cur_frame_pose['pose_keypoints_2d'][7*3:7*3 + 2]
                if (np.array(cur_frame_pose['pose_keypoints_2d'][4*3:4*3 + 2]) -
                    np.array(cur_frame_pose['hand_right_keypoints_2d'][0:2])).max() > 15 and \
                        cur_frame_pose['pose_keypoints_2d'][4*3 + 2] > 0.2:
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

    if datum_id in PJM_LEFT_VIDEOS_LST:
        pose = flip_pose(pose)

    normalization_info = pose_normalization_info(pose_header)
    pose = pose.normalize(normalization_info)
    pose_hide_legs(pose)
    pose_hide_low_conf(pose)

    # Prune all leading frames containing only zeros, almost no face, or no hands
    for i in range(len(pose.body.data)):
        if pose.body.confidence[i][:, 25:-42].sum() > 35 and \
                pose.body.confidence[i][:, 4] + pose.body.confidence[i][:, 7] > 0:
            if i != 0:
                pose.body.data = pose.body.data[i:]
                pose.body.confidence = pose.body.confidence[i:]
            break

    # Prune all trailing frames containing only zeros, almost no face, or no hands
    for i in range(len(pose.body.data) - 1, 0, -1):
        if pose.body.confidence[i][:, 25:-42].sum() > 35 and \
                pose.body.confidence[i][:, 4] + pose.body.confidence[i][:, 7] > 0:
            if i != len(pose.body.data) - 1:
                pose.body.data = pose.body.data[:i + 1]
                pose.body.confidence = pose.body.confidence[:i + 1]
            break

    return pose


# def count_spread_fingers(pose_hand, th=0.1):
#     spread_fingers_count = 0
#     finger_points = [(0,17,18,19,20), (0,13,14,15,16), (0,9,10,11,12), (0,5,6,7,8), (0,1,2,3,4)]
#     for pt_group in finger_points:
#         pts = [pose_hand[:, i] for i in pt_group]
#         var = np.var(pts)
#         if var < th:
#             spread_fingers_count += 1
#     return spread_fingers_count


def measure_distance(pose1_id, pose2_id, keypoints_path):
    pose1 = get_pose(os.path.join(keypoints_path, pose1_id), pose1_id)
    pose2 = get_pose(os.path.join(keypoints_path, pose2_id), pose2_id)
    seq_lens_diff = np.abs(pose1.body.data.shape[0]-pose2.body.data.shape[0])
    # print("seq len diff:", seq_lens_diff)

    pose1_body = pose1.body.data.squeeze(1)
    pose2_body = pose2.body.data.squeeze(1)

    if pose1.body.data.shape[0] < pose2.body.data.shape[0]:
        pose1_body = np.ma.concatenate((pose1_body, np.zeros((seq_lens_diff, 137, 2))))
    if pose2.body.data.shape[0] < pose1.body.data.shape[0]:
        pose2_body = np.ma.concatenate((pose2_body, np.zeros((seq_lens_diff, 137, 2))))

    sum_pose_diff = np.sum((pose1_body - pose2_body) ** 2)
    mean_pose_diff = np.mean((pose1_body - pose2_body) ** 2)
    # print("sum pose diff:", sum_pose_diff)
    # print("mean pose diff:", mean_pose_diff)

    pose1_frame_diff = np.array([np.sign(pose1_body[i] - pose1_body[i + 1]) for i in
                              range(len(pose1_body) - 1)])
    pose2_frame_diff = np.array([np.sign(pose2_body[i] - pose2_body[i + 1]) for i in
                                 range(len(pose2_body) - 1)])

    sum_frame_diff_diff = np.sum((pose1_frame_diff-pose2_frame_diff)**2)
    mean_frame_diff_diff = np.mean((pose1_frame_diff-pose2_frame_diff)**2)
    # print("sum frame diff diff:", sum_frame_diff_diff)
    # print("mean frame diff diff:", mean_frame_diff_diff)
    # print()
    pose1_lhand = pose1_body[:, -42:-21]
    pose1_rhand = pose1_body[:, -21:]
    pose2_lhand = pose2_body[:, -42:-21]
    pose2_rhand = pose2_body[:, -21:]
    sum_lhand_diff = np.sum((pose1_lhand - pose2_lhand) ** 2)
    mean_lhand_diff = np.mean((pose1_lhand - pose2_lhand) ** 2)
    sum_rhand_diff = np.sum((pose1_rhand - pose2_rhand) ** 2)
    mean_rhand_diff = np.mean((pose1_rhand - pose2_rhand) ** 2)
    # spread_fingers1_lhand = count_spread_fingers(pose1_lhand)
    # spread_fingers1_rhand = count_spread_fingers(pose1_rhand)
    # spread_fingers2_lhand = count_spread_fingers(pose2_lhand)
    # spread_fingers2_rhand = count_spread_fingers(pose2_rhand)

    return (mean_rhand_diff + mean_lhand_diff) / 2


def get_keypoint_trajectory(pose, keypoint_idx, pose_id, plot=False):
    normalization_info = pose_normalization_info(pose.header)
    pose = pose.normalize(normalization_info, scale_factor=100)
    pose.focus()
    height = pose.header.dimensions.height
    all_pts = pose.body.data[:, :, keypoint_idx].squeeze(1)
    poly = np.polyfit(all_pts[:, 0], all_pts[:, 1], len(all_pts)*3//4)
    # normalized_pts = all_pts-all_pts[0]
    if plot:
        for pt in range(len(all_pts)-1):
            if np.ma.is_masked(all_pts[pt]):
                continue
            plt.plot(int(all_pts[pt][0]), height-int(all_pts[pt][1]))
            if np.ma.is_masked(all_pts[pt+1]):
                continue
            plt.arrow(int(all_pts[pt][0]), height-int(all_pts[pt][1]), int(all_pts[pt+1][0])-int(all_pts[pt][0]),
                      -(int(all_pts[pt+1][1])-int(all_pts[pt][1])), head_width=1)
        if not np.ma.is_masked(all_pts[-1]):
            plt.plot(int(all_pts[-1][0]), height-int(all_pts[-1][1]))
        plt.title(f"route of keypoint {keypoint_idx} for {pose_id}")
        plt.show()

    return poly


def EDR_distance(trajectory1, trajectory2, th=0.1):
    EDR = np.full((len(trajectory1), len(trajectory2)), np.infty)
    EDR[0, 0] = 0

    for i in range(len(trajectory1)):
        for j in range(len(trajectory2)):
            if np.ma.is_masked(trajectory1[i]) and not np.ma.is_masked(trajectory2[j]):
                min_cost = len(trajectory2)
            elif not np.ma.is_masked(trajectory1[i]) and np.ma.is_masked(trajectory2[j]):
                min_cost = len(trajectory1)
            elif np.ma.is_masked(trajectory1[i]) and np.ma.is_masked(trajectory2[j]):
                min_cost = max(len(trajectory1), len(trajectory2))
            else:
                distance = np.sqrt(
                    (trajectory1[i][0] - trajectory2[j][0]) ** 2 + (trajectory1[i][1] - trajectory2[j][1]) ** 2)
                if i != 0 and j != 0:
                    penalty = 0 if distance < th else 1
                    min_cost = np.min([EDR[i - 1, j] + 1,  # insertion
                                       EDR[i, j - 1] + 1,  # deletion
                                       EDR[i - 1, j - 1] + penalty])  # match
                elif j != 0:
                    min_cost = EDR[i, j - 1] + 1
                elif i != 0:
                    min_cost = EDR[i - 1, j] + 1
                else:
                    min_cost = 0

            EDR[i, j] = min_cost

    return EDR[-1, -1]


def DTWdistance(trajectory1, trajectory2):
    DTW = np.full((len(trajectory1), len(trajectory2)), np.infty)
    DTW[0, 0] = 0

    for i in range(len(trajectory1)):
        for j in range(len(trajectory2)):
            if np.ma.is_masked(trajectory1[i]) and not np.ma.is_masked(trajectory2[j]):
                cost = 0#np.inf#trajectory2[j]
            elif not np.ma.is_masked(trajectory1[i]) and np.ma.is_masked(trajectory2[j]):
                cost = 0#np.inf#trajectory1[i]
            elif np.ma.is_masked(trajectory1[i]) and np.ma.is_masked(trajectory2[j]):
                cost = 0
            else:
                cost = np.sqrt((trajectory1[i][0]-trajectory2[j][0])**2+(trajectory1[i][1]-trajectory2[j][1])**2)
            if i != 0 and j != 0:
                min_cost = np.min([DTW[i-1, j],    # insertion
                                  DTW[i, j-1],    # deletion
                                  DTW[i-1, j-1]])    # match
            elif j != 0:
                min_cost = DTW[i, j - 1]  # deletion
            elif i != 0:
                min_cost = DTW[i - 1, j]  # insertion
            else:
                min_cost = 0

            DTW[i, j] = cost + min_cost

    return DTW[-1, -1]


def masked_euclidean(point1, point2):
    if np.ma.is_masked(point1) or np.ma.is_masked(point2):
        return 0
    # if np.ma.is_masked(point2):
    #     return 100
    d = euclidean(point1, point2)
    return d


def masked_mse(trajectory1, trajectory2, confidence):
    if len(trajectory1) < len(trajectory2):
        diff = len(trajectory2) - len(trajectory1)
        trajectory1 = np.concatenate((trajectory1, np.zeros((diff, 2))))
        confidence = np.concatenate((confidence, np.zeros((diff))))
    elif len(trajectory2) < len(trajectory1):
        trajectory2 = np.concatenate((trajectory2, np.zeros((len(trajectory1) - len(trajectory2), 2))))
    sq_error = np.power(trajectory1 - trajectory2, 2).sum(-1)
    return (sq_error * confidence).mean()


def compare_pose_videos(pose1_id, pose2_id, keypoints_path, distance_function=fastdtw):
    pose1 = get_pose(os.path.join(keypoints_path, pose1_id), pose1_id)
    pose2 = get_pose(os.path.join(keypoints_path, pose2_id), pose2_id)
    return compare_poses(pose1, pose2, distance_function)


def compare_poses(pose1, pose2, distance_function=fastdtw):
    # don't use legs, face for trajectory distance computations- only upper body and hands
    pose1_data_normalized = np.ma.concatenate([pose1.body.data[:, :, :95] - pose1.body.data[0, 0, 0],
                                            pose1.body.data[:, :, 95:116] - pose1.body.data[0, 0, 95],
                                            pose1.body.data[:, :, 116:] - pose1.body.data[0, 0, 116]], axis=2)
    pose2_data_normalized = np.ma.concatenate([pose2.body.data[:, :, :95] - pose2.body.data[0, 0, 0],
                                            pose2.body.data[:, :, 95:116] - pose2.body.data[0, 0, 95],
                                            pose2.body.data[:, :, 116:] - pose2.body.data[0, 0, 116]], axis=2)

    total_distance = 0
    idx2weight = {i: 0.2 for i in range(9)}
    idx2weight.update({i: 0.8 for i in range(95, pose1.body.data.shape[2])})
    for keypoint_idx, weight in idx2weight.items():
        pose1_keypoint_trajectory = pose1_data_normalized[:, :, keypoint_idx].squeeze(1)
        pose2_keypoint_trajectory = pose2_data_normalized[:, :, keypoint_idx].squeeze(1)
        if np.ma.is_masked(pose1_keypoint_trajectory) or np.ma.is_masked(pose2_keypoint_trajectory):
            continue
        if distance_function == masked_mse:
            dist = distance_function(pose1_keypoint_trajectory, pose2_keypoint_trajectory, pose1.body.confidence[:,
                                                                                           :, keypoint_idx].squeeze(1))
        else:
            dist = distance_function(pose1_keypoint_trajectory, pose2_keypoint_trajectory, dist=masked_euclidean)[0]
        total_distance += dist*weight
    return total_distance/len(idx2weight)


def visualize_candidates(poses):  # TODO- doesn't work at the moment
    from mpl_toolkits.axes_grid1 import ImageGrid
    from pose_format.pose_visualizer import PoseVisualizer

    frames = np.zeros_like(poses)
    for i in range(len(poses)):
        # Normalize pose
        normalization_info = pose_normalization_info(poses[i].header)
        pose = poses[i].normalize(normalization_info, scale_factor=100)
        pose.focus()
        visualizer = PoseVisualizer(pose, thickness=2)
        pose_frames = list(visualizer.draw())
        frames[i] = pose_frames

    image_size = (max([frame[0].shape[1] for frame in frames]) * len(poses),
                  max([frame[0].shape[0] for frame in frames]))
    # out = cv2.VideoWriter("metric_test.mp4", cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 25, image_size)
    max_seq_len = np.max([len(seq) for seq in frames])
    for i in range(max_seq_len):
        fig = plt.figure(figsize=image_size)
        grid = ImageGrid(fig, 111,  # similar to subplot(111)
                         nrows_ncols=(4, 4),  # creates 4x4 grid of axes
                         axes_pad=0.1,  # pad between axes in inch.
                         )
        for ax, pose in zip(grid, frames):
            if i > len(pose):
                ax.imshow(np.full_like(pose[0]), 255)
            else:
                ax.imshow(pose[i])
        plt.show()
        h = 0
        # out.write(grid_frame)
    # out.release()

    return frames


def calc_fid_score(data1_features, data2_features):
    # calculate mean and covariance statistics over the feature vectors
    mu1 = data1_features.mean(axis=0)
    sigma1 = cov(data1_features.cpu(), rowvar=False)
    mu2 = data2_features.mean(axis=0)
    sigma2 = cov(data2_features.cpu(), rowvar=False)

    # calculate sum squared difference between means
    ssdiff = torch.pow(mu1 - mu2, 2).sum(-1) #np.sum((mu1 - mu2) ** 2.0)
    # calculate sqrt of product between cov
    covmean = sqrtm(sigma1.dot(sigma2))
    # check and correct imaginary numbers from sqrt
    if iscomplexobj(covmean):
        covmean = covmean.real
    # calculate score
    fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid


def get_mean_pose_features(model, dataloader):
    mean_pose_embedding = torch.Tensor().to(model.device)
    for batch in dataloader:
        mean_embedded_pose = model.embed_pose(batch["pose"]["data"].to(model.device)).mean(axis=1)
        mean_pose_embedding = torch.cat((mean_pose_embedding, mean_embedded_pose))
    return mean_pose_embedding


def get_all_preds_features(model, dataset):
    pred_mean_features = torch.Tensor().to(model.device)
    for i, datum in enumerate(dataset):
        first_pose = datum["pose"]["data"][0].to(model.device)
        seq_iter = model.forward(text=datum["text"], first_pose=first_pose)
        for i in range(model.num_steps):
            seq = next(seq_iter)

        mean_embedded_pose = model.embed_pose(seq.unsqueeze(0)).mean(axis=1)
        pred_mean_features = torch.cat((pred_mean_features, mean_embedded_pose))
    return pred_mean_features


def get_fid_scores(model, train_loader, validation_loader, train_dataset, validation_dataset):
    model.eval()
    with torch.no_grad():
        train_features = get_mean_pose_features(model, train_loader)
        val_features = get_mean_pose_features(model, validation_loader)

        fid_score = calc_fid_score(train_features, train_features)
        print("FID score train features to themselves", fid_score.item())
        fid_score = calc_fid_score(train_features, val_features)
        print("FID score train and val features", fid_score.item())

        train_pred_features = get_all_preds_features(model, train_dataset)
        val_pred_features = get_all_preds_features(model, validation_dataset)

        fid_score = calc_fid_score(train_features, train_pred_features)
        print("FID score train label and pred features", fid_score.item())
        fid_score = calc_fid_score(train_features, val_pred_features)
        print("FID score train label and val pred features", fid_score.item())
        fid_score = calc_fid_score(val_features, val_pred_features)
        print("FID score val label and pred features", fid_score.item())


def get_hamnosys_similarities(pickle_data_path="/home/nlp/rotemsh/slt/data/pose_data/cnn_data_dgs.train"):
    with open(pickle_data_path, 'rb') as f:
        train_data = pickle.load(f)

    all_vectors = [datum["hamnosys_vector"].numpy() for datum in train_data]
    from sklearn.metrics.pairwise import cosine_similarity

    all_sim = cosine_similarity(all_vectors, all_vectors)
    for sim_vec in all_sim:
        print("-------------------------------------")
        for i in np.where(sim_vec > 0.8)[0]:
            print(train_data[i]["gloss"])
            print(train_data[i]["hamnosys"])
            print(sim_vec[i])


def __compare_pred_to_video(pred, keypoints_path, pose_id, distance_function):
    label_pose = get_pose(os.path.join(keypoints_path, pose_id), pose_id)
    pose1_data_normalized = np.ma.concatenate([pred.body.data[:, :, :95] - pred.body.data[0, 0, 0],
                                            pred.body.data[:, :, 95:116] - pred.body.data[0, 0, 95],
                                            pred.body.data[:, :, 116:] - pred.body.data[0, 0, 116]], axis=2)
    pose2_data_normalized = np.ma.concatenate([label_pose.body.data[:, :, :95] - label_pose.body.data[0, 0, 0],
                                            label_pose.body.data[:, :, 95:116] - label_pose.body.data[0, 0, 95],
                                            label_pose.body.data[:, :, 116:] - label_pose.body.data[0, 0, 116]], axis=2)

    total_distance = 0
    idx2weight = {i: 0.2 for i in range(9)}
    idx2weight.update({i: 1 for i in range(95, pred.body.data.shape[2])})
    for keypoint_idx, weight in idx2weight.items():
        # don't use legs, face for trajectory distance computations- only upper body and hands
        pose1_keypoint_trajectory = pose1_data_normalized[:, :, keypoint_idx].squeeze(1)
        pose2_keypoint_trajectory = pose2_data_normalized[:, :, keypoint_idx].squeeze(1)
        dist = distance_function(pose1_keypoint_trajectory, pose2_keypoint_trajectory, dist=masked_euclidean)[0]
        total_distance += dist * weight
    return total_distance / len(idx2weight)


def check_ranks(distances, index):
    rank_1 = (index == distances[0])
    rank_5 = (index in distances[:5])
    rank_10 = (index in distances)
    return rank_1, rank_5, rank_10


def get_poses_ranks(pred, pred_id, keypoints_path, data_ids, distance_function=fastdtw, num_samples=30):
    pred2label_distance = __compare_pred_to_video(pred, keypoints_path, pred_id, distance_function)
    # print(f"\ndistance between pred and label for {pred_id} is {pred2label_distance}")

    distances_to_label = [pred2label_distance]
    distances_to_pred = [pred2label_distance]
    pred2label_index = 0

    pose_ids = random.sample(data_ids, num_samples)
    for i, pose_id in enumerate(pose_ids):
        distances_to_pred.append(__compare_pred_to_video(pred, keypoints_path, pose_id, distance_function))
        distances_to_label.append(compare_pose_videos(pose_id, pred_id, keypoints_path, distance_function))

    # pose_ids = [pred_id] + pose_ids
    # print("\n10 best sorted distances to pred")
    best_pred = np.argsort(distances_to_pred)[:10]
    # for i in best_pred:
        # print(f"{pose_ids[i]}, {distances_to_pred[i]}")
    rank_1_pred, rank_5_pred, rank_10_pred = check_ranks(best_pred, pred2label_index)
    # print("\n10 best sorted distances to label")
    best_label = np.argsort(distances_to_label)[:10]
    # for i in best_label:
    #     print(f"{pose_ids[i]}, {distances_to_label[i]}")
    rank_1_label, rank_5_label, rank_10_label = check_ranks(best_label, pred2label_index)

    return pred2label_distance, rank_1_pred, rank_5_pred, rank_10_pred, rank_1_label, rank_5_label, rank_10_label


if __name__ == "__main__":
    # vids = ["10262", "70224", "70222", "24015", "10569", "10529"]
    signer1_vids = ["87846", "87383", "87387", "87855", "88638", "89318", "94851", "80997"]
    from itertools import combinations
    keypoints_path = "/home/nlp/rotemsh/SLP/data/keypoints_dir"
    # poses = [get_pose(os.path.join(keypoints_path, pose_id), pose_id) for pose_id in signer1_vids]
    # from text_to_pose.pred import visualize_pose
    # visualize_candidates(poses)
    distances = []
    # pairs = list(combinations(vids, 2))
    for i in range(1, len(signer1_vids)):
        # visualize_pose([poses[0], poses[i]], f"metric_test_{i}.mp4", "videos/metrics_tests")
        dist = compare_pose_videos(signer1_vids[0], signer1_vids[i], keypoints_path, fastdtw)
        distances.append(dist)
        # print(f"distance between {signer1_vids[0]} and {signer1_vids[i]} is {dist}")

    for i in np.argsort(distances):
        print(f"distance between {signer1_vids[0]}, {signer1_vids[i+1]}, {distances[i]}")

    # for vid in vids:
    #     pose = get_pose(os.path.join(keypoints_path, vid), vid)

        # print(f"distance between {vid1} and {vid2}")
        # distances.append(measure_distance(vid1, vid2, keypoints_path))


