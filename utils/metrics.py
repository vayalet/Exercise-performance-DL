from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from matplotlib import pyplot as plt
import numpy as np
import random
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr
import os
from data.loader import *
import h5py

#from models import models
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm


sns.set(style="whitegrid")

def get_data(data, task):
    if task == 'class':
        pred =  [int(x) for x in data[:, 0]]
        true =  [int(x) for x in data[:, 1]]
    else: 
        pred =  data[:, :17, :]
        true =  data[:, 17:, :]
    return pred, true

def get_labels_start_end_time(frame_wise_labels):
    labels = []
    starts = []
    ends = []
    last_label = frame_wise_labels[0]
    labels.append(frame_wise_labels[0])
    starts.append(0)
    for i in range(len(frame_wise_labels)):
        if frame_wise_labels[i] != last_label:
            labels.append(frame_wise_labels[i])
            starts.append(i)
            ends.append(i)
            last_label = frame_wise_labels[i]
    ends.append(i + 1)
    return labels, starts, ends

def segment_f1_score(recognized, ground_truth, overlap=0.5):
    p_label, p_start, p_end = get_labels_start_end_time(recognized)
    y_label, y_start, y_end = get_labels_start_end_time(ground_truth)

    tp = 0
    fp = 0

    hits = np.zeros(len(y_label))
    if len(y_label) == 0:
        fp = len(p_label)
    else:
        for j in range(len(p_label)):
            intersection = np.minimum(p_end[j], y_end) - np.maximum(p_start[j], y_start)
            union = np.maximum(p_end[j], y_end) - np.minimum(p_start[j], y_start)
            #print(intersection, union)
            IoU = (1.0 * intersection / union) * ([p_label[j] == y_label[x] for x in range(len(y_label))])
            # Get the best scoring segment
            idx = np.array(IoU).argmax()
            #print(IoU)

            if IoU[idx] >= overlap and not hits[idx]:
                tp += 1
                hits[idx] = 1
            else:
                fp += 1
    fn = len(y_label) - sum(hits)
    precision = tp / (tp+fp) if (tp+fp)>0 else 0.0
    recall = tp / (tp+fn) if (tp+fn)>0 else 0.0
    f1 = 2 * precision * recall / (precision+recall) if (precision+recall)>0 else 0.0

    return f1

def performance_metrics(actual, predicted):
    accuracy = np.round((accuracy_score(actual, predicted)), 4)
    f1 = np.round((f1_score(actual, predicted, average='weighted')), 4)
    precision = np.round((precision_score(actual, predicted, average='weighted')), 4)
    recall = np.round((recall_score(actual, predicted, average='weighted')), 4)
    segment_f1 = np.round((segment_f1_score(actual, predicted, 0.5)), 4)

    print("Weighted average metrics:")
    print("Accuracy:", accuracy)
    print("F1-Score:", f1)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1@50:", segment_f1)

    return accuracy, f1, precision, recall, segment_f1

import numpy as np

def compute_mae(pred_angles, true_angles, joint_mask, mode="overall"):
    """
    Compute the Mean Angular Error (MAE) in different modes: overall, per coordinate, or per joint.

    Parameters:
    - pred_angles: np.ndarray of shape (batch_size, num_frames, num_joints, 3), in degrees
    - true_angles: np.ndarray of shape (batch_size, num_frames, num_joints, 3), in degrees
    - joint_mask: np.ndarray of shape (num_joints,), mask for joints to include in the calculation
    - mode: str, specifies which MAE to compute:
        - "overall": Mean Angular Error across all joints, frames, and coordinates.
        - "per_coordinate": MAE for each coordinate (x, y, z).
        - "per_joint": MAE for each joint across all frames and coordinates.

    Returns:
    - MAE in the specified mode:
        - "overall": float
        - "per_coordinate": np.ndarray of shape (3,) (x, y, z)
        - "per_joint": np.ndarray of shape (num_joints,)
    """
    assert pred_angles.shape == true_angles.shape, "Shape mismatch between predicted and true angles."

    # Compute absolute angular difference
    angle_diff = np.abs(pred_angles - true_angles)

    # Normalize angular differences to be within [0, 180] degrees
    angle_diff = np.where(angle_diff > 180, 360 - angle_diff, angle_diff)

    # Reshape joint_mask for broadcasting across batch, frames, and angular components
    joint_mask = joint_mask.reshape(1, 1, -1, 1)  # Shape: (1, 1, num_joints, 1)

    # Apply joint mask
    masked_angle_diff = angle_diff * joint_mask

    if mode == "overall":
        # Total number of active elements (masked joints only)
        total_active_elements = np.sum(joint_mask) * pred_angles.shape[0] * pred_angles.shape[1]
        # Compute overall MAE
        mae = np.sum(masked_angle_diff) / (total_active_elements * pred_angles.shape[-1])
        return mae

    elif mode == "coord":
        # Total number of active elements per joint per coordinate
        total_active_elements = (
            np.sum(joint_mask, axis=-1, keepdims=True) * pred_angles.shape[0] * pred_angles.shape[1]
        )  # Shape: (num_joints, 1)

        # Prevent division by zero by setting invalid results to 0
        total_active_elements = np.where(total_active_elements == 0, 1, total_active_elements)

        # Compute MAE per joint per coordinate
        mae_joint_coord = np.sum(masked_angle_diff, axis=(0, 1)) / total_active_elements
        mae_joint_coord[joint_mask.squeeze(-1) == 0] = 0  # Set masked joints to 0
        return mae_joint_coord  # Shape: (num_joints, 3)

    elif mode == "joint":
        # Total number of active elements per joint
        total_active_elements = (
            np.sum(joint_mask, axis=-1) * pred_angles.shape[0] * pred_angles.shape[1] * pred_angles.shape[-1]
        )
        total_active_elements = np.where(total_active_elements == 0, 1, total_active_elements)

        # Sum angular differences for each joint across all frames and coordinates
        mae_joints = np.sum(masked_angle_diff, axis=(0, 1, 3)) / total_active_elements
        return mae_joints

    elif mode == "mean_per_coord":
        # Total number of active elements across all joints per coordinate
        total_active_elements = np.sum(joint_mask) * pred_angles.shape[0] * pred_angles.shape[1]

        # Compute the mean MAE per coordinate across all non-masked joints
        mae_per_coord = np.sum(masked_angle_diff, axis=(0, 1, 2)) / (total_active_elements)
        return mae_per_coord

    elif mode == "joint_no_z":
        # Exclude Z coordinate by slicing and summing only X and Y (axis 2)
        masked_angle_diff_no_z = masked_angle_diff[..., :2]  # Shape: (..., 2)

        # Total number of active elements per joint excluding Z
        total_active_elements = (
            np.sum(joint_mask, axis=-1) * pred_angles.shape[0] * pred_angles.shape[1] * 2
        )  # 2 corresponds to the X and Y dimensions
        total_active_elements = np.where(total_active_elements == 0, 1, total_active_elements)

        # Compute MAE per joint excluding Z
        mae_joints_no_z = np.sum(masked_angle_diff_no_z, axis=(0, 1, 3)) / total_active_elements
        return mae_joints_no_z

def compute_mae_videopose(pred_angles, true_angles, joint_mask, mode="overall"):
    """
    Compute the Mean Angular Error (MAE) or Root Mean Squared Error (RMSE) for data 
    in shape (n_frames, num_joints, 3).

    Parameters:
    - pred_angles: np.ndarray of shape (n_frames, num_joints, 3), in degrees
    - true_angles: np.ndarray of shape (n_frames, num_joints, 3), in degrees
    - joint_mask: np.ndarray of shape (num_joints,), mask for joints to include in the calculation
    - mode: str, one of:
        - "overall", "per_coordinate", "per_joint"
        - "overall_rmse", "per_coordinate_rmse", "per_joint_rmse"
        - "mean_per_coord", "joint_no_z"

    Returns:
    - Metric in the specified mode
    """
    assert pred_angles.shape == true_angles.shape, "Shape mismatch between predicted and true angles."

    # Compute angular difference
    angle_diff = np.abs(pred_angles - true_angles)
    angle_diff = np.where(angle_diff > 180, 360 - angle_diff, angle_diff)

    joint_mask = joint_mask.reshape(1, -1, 1)
    masked_angle_diff = angle_diff * joint_mask
    squared_diff = (angle_diff ** 2) * joint_mask

    if mode == "overall":
        total_elements = np.sum(joint_mask) * pred_angles.shape[0]
        return np.sum(masked_angle_diff) / (total_elements * pred_angles.shape[-1])

    elif mode == "overall_rmse":
        total_elements = np.sum(joint_mask) * pred_angles.shape[0]
        mse = np.sum(squared_diff) / (total_elements * pred_angles.shape[-1])
        return np.sqrt(mse)

    elif mode == "per_coordinate":
        total_elements = np.sum(joint_mask) * pred_angles.shape[0]
        return np.sum(masked_angle_diff, axis=(0, 1)) / total_elements

    elif mode == "per_coordinate_rmse":
        total_elements = np.sum(joint_mask) * pred_angles.shape[0]
        mse_per_coord = np.sum(squared_diff, axis=(0, 1)) / total_elements
        return np.sqrt(mse_per_coord)

    elif mode == "per_joint":
        total_elements = (
            np.sum(joint_mask, axis=-1) * pred_angles.shape[0] * pred_angles.shape[-1]
        )
        total_elements = np.where(total_elements == 0, 1, total_elements)
        return np.sum(masked_angle_diff, axis=(0, 2)) / total_elements

    elif mode == "per_joint_per_coord":
        # Get number of frames and joint dims
        n_frames, n_joints, n_coords = pred_angles.shape

        # Broadcast mask to full shape: (n_frames, n_joints, n_coords)
        full_mask = joint_mask.reshape(1, -1, 1).repeat(n_frames, axis=0).repeat(n_coords, axis=2)

        # Compute number of active (masked-in) elements per joint and coordinate
        total_elements = np.sum(full_mask, axis=0)  # shape: (n_joints, 3)
        total_elements = np.where(total_elements == 0, 1, total_elements)  # avoid divide-by-zero

        # Sum angular errors per joint per coordinate
        mae_per_joint_coord = np.sum(masked_angle_diff, axis=0) / total_elements  # shape: (n_joints, 3)

        return mae_per_joint_coord

    elif mode == "per_joint_rmse":
        total_elements = (
            np.sum(joint_mask, axis=-1) * pred_angles.shape[0] * pred_angles.shape[-1]
        )
        total_elements = np.where(total_elements == 0, 1, total_elements)
        mse_per_joint = np.sum(squared_diff, axis=(0, 2)) / total_elements
        return np.sqrt(mse_per_joint)

    elif mode == "mean_per_coord":
        total_elements = np.sum(joint_mask) * pred_angles.shape[0]
        return np.sum(masked_angle_diff, axis=(0, 1)) / total_elements

    elif mode == "joint_no_z":
        masked_angle_diff_no_z = masked_angle_diff[..., :2]
        total_elements = np.sum(joint_mask, axis=-1) * pred_angles.shape[0] * 2
        total_elements = np.where(total_elements == 0, 1, total_elements)
        return np.sum(masked_angle_diff_no_z, axis=(0, 2)) / total_elements

    else:
        raise ValueError(f"Invalid mode: {mode}")
def compute_correlation(pred, actual):
    pred_flat = pred.reshape(-1)
    actual_flat = actual.reshape(-1)

    correlation, _ = pearsonr(pred_flat, actual_flat)
    return correlation