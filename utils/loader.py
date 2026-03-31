import math
import random
from sklearn.base import TransformerMixin
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, LabelEncoder
from torch.utils.data import Dataset, DataLoader, TensorDataset, ConcatDataset
import matplotlib.pyplot as plt
import scipy
import statistics as st
import os
import numpy as np
import pandas as pd
import sys
from collections import Counter
import h5py

import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset, ConcatDataset

def get_mb_data(path, step_size, window_len):
    with h5py.File(path, 'r') as f:
    # Load the dataset from the HDF5 file
        data = f['representations'][:]
    
    return data



from scipy.stats import mode

class FullSequenceDataset(Dataset):
    """
    Load the AlphaPose data for a single subject for VideoPose3D testing.
    """
    def __init__(self, subjects, test=False, task="classification"):
        self.subjects = subjects
        self.test = test
        self.task = task
        self.skip_cols = [
            'Frame', 'Right Ball Foot Abduction/Adduction', 'Right Ball Foot Internal/External Rotation',
            'Right Ball Foot Flexion/Extension', 'Left Ball Foot Abduction/Adduction',
            'Left Ball Foot Internal/External Rotation', 'Left Ball Foot Flexion/Extension',
            'C1 Head Lateral Bending', 'C1 Head Axial Rotation', 'C1 Head Flexion/Extension',
            'L4L3 Lateral Bending', 'L4L3 Axial Rotation', 'L4L3 Flexion/Extension',
            'L1T12 Lateral Bending', 'L1T12 Axial Rotation', 'L1T12 Flexion/Extension'
        ]
        self.subject_data = self._load_subject_data()

    def _load_subject_data(self):
        """
        Load and process data for each subject as a full sequence.
        """
        base_path_data = '.../data/test/' if self.test else 'C:/Users/u0170757/OneDrive - KU Leuven/Documents/AI@WZC/Dataset/'
        path_imu = '.../results_test_pose/test/' if self.test else 'C:/Users/u0170757/OneDrive - KU Leuven/Documents/AI@WZC/Dataset/2D_Pose/'

        all_data = []
        for subject in self.subjects:
            print(f"Processing subject {subject}")
            subject_data = []
            subject_poses = []
            subject_labels = []
            train_filenames = []

            # Discover .xlsx files for the subject
            for dirpath, _, filenames in os.walk(base_path_data + '/Joint_angles/' + subject):
                for filename in [f for f in filenames if f.endswith(".xlsx")]:
                    train_filenames.append(filename)

            # Process each discovered .xlsx file
            for filename in train_filenames:
                print(f"Loading data for file: {filename}")
                
                # Load AlphaPose data (.npy) for the file
                data = np.load(path_imu + subject + '/AlphaPose_' + filename[:-5] + '.npy', allow_pickle=True)

                # Load the corresponding labels
                label = np.loadtxt(base_path_data + '/Exercise_labels/'+ subject + '/' + filename[:-5] + '_labels.csv')

                # Mark and exclude double frames in labels
                # double_frames = np.load(base_path_data + '/Joint_angles/'+ subject + '/AlphaPose_' + filename[:-5] + '.npy')
                # for frame in double_frames:
                #     idx = int(frame[:-4])
                #     label[idx] = 99

                # Load pose data from Excel, excluding specified columns
                pose = pd.read_excel(
                    f"{base_path_data}/Joint_angles/{subject}/{filename}", 
                    sheet_name="Joint Angles XZY", 
                    usecols=lambda col: col not in self.skip_cols
                ).to_numpy()

                # Align data and label lengths
                min_len = min(len(data), len(label), len(pose))
                data, label, pose = data[:min_len], label[:min_len], pose[:min_len]

                # Filter out invalid frames (where label == 99)
                valid_indices = label != 99
                data, label, pose = data[valid_indices], label[valid_indices], pose[valid_indices]
                print(f" - Data, label, and pose shapes after filtering: {data.shape}, {label.shape}, {pose.shape}")

                # Reshape 3D pose data into (num_frames, 17, 3)
                pose = pose.reshape(pose.shape[0], 17, 3)

                window_len = 243
                step_size = 1
                valid_data = []
                valid_pose = []

                for start_idx in range(0, len(data) - window_len + 1, step_size):
                    end_idx = start_idx + window_len
                    if end_idx > len(data):
                        break  # Incomplete window at the end
                    center_idx = start_idx + window_len // 2
                    valid_data.append(data[center_idx])
                    valid_pose.append(pose[center_idx])

                if valid_data:
                    subject_data.append(np.stack(valid_data))
                    subject_poses.append(np.stack(valid_pose))
            # Concatenate sequences for the subject
            if subject_data:
                if self.task == "regression":
                    all_data.append({
                        'pose_2d': np.concatenate(subject_data, axis=0),
                        'pose_3d': np.concatenate(subject_poses, axis=0)
                    })
                else:
                     all_data.append({
                        'pose_2d': np.concatenate(subject_data, axis=0),
                        'labels': np.concatenate(subject_labels, axis=0)
                    })

        return all_data

    def __len__(self):
        return len(self.subject_data)

    def __getitem__(self, idx):
        """
        Fetch the full sequence for the subject.
        """
        sample = self.subject_data[idx]
        X = sample['pose_2d']  # 2D keypoints
        if self.task == "regression":
            y = sample['pose_3d']  # 3D ground-truth poses
        else:
            y = sample['labels']
        pad = 121
        X = np.pad(X,
                    ((pad, pad), (0, 0), (0, 0)),
                    'edge')
        if self.task == "classification":
            
            return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.int32)
        else:
            return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

def map_labels_to_exercises(labels):
    label_map = {
        0: 0,  # Random movements
        1: 1,  # Sit-to-stand
        2: 1,
        3: 2,  # Bending forward/backward on chair
        4: 2,
        5: 3,  # Moving a weight between armrests
        6: 3,
        7: 4,  # Lateral leg movement
        8: 4,
        9: 5,  # Picking up a box
        10: 5,
        11: 6,  # Touching a cup with foot
        12: 6,
        13: 6,
        14: 6
    }
    return [label_map.get(int(l), l) for l in labels]
    
class APDataset(Dataset):
    """
    Load the AlphaPose data.
    """
    def __init__(self, subjects, window_len, step_size=1, test=False, task="classification"):
        self.subjects = subjects
        self.window_len = window_len
        self.step_size = step_size
        self.test = test
        self.task = task
        self.skip_cols = [
            'Frame', 'Right Ball Foot Abduction/Adduction', 'Right Ball Foot Internal/External Rotation', 'Right Ball Foot Flexion/Extension',
            'Left Ball Foot Abduction/Adduction', 'Left Ball Foot Internal/External Rotation', 'Left Ball Foot Flexion/Extension',
            'C1 Head Lateral Bending', 'C1 Head Axial Rotation', 'C1 Head Flexion/Extension',
            'L4L3 Lateral Bending', 'L4L3 Axial Rotation', 'L4L3 Flexion/Extension',
            'L1T12 Lateral Bending', 'L1T12 Axial Rotation', 'L1T12 Flexion/Extension'
        ]
        self.subject_data = self._load_subject_data()

    def _load_subject_data(self):
        """
        Load and process data for each subject, generating sliding windows and checking data integrity.
        """
        base_path_data = '.../data/test/' if self.test else 'C:/Users/u0170757/OneDrive - KU Leuven/Documents/AI@WZC/Dataset/'
        path_imu = '.../results_test_pose/test/' if self.test else 'C:/Users/u0170757/OneDrive - KU Leuven/Documents/AI@WZC/Dataset/2D_Pose/'

        all_poses = []
        all_windows = []
        for subject in self.subjects:
            print(f"Processing subject {subject}")
            subject_windows = []
            subject_labels = []
            subject_poses = []
            train_filenames = []

            # Discover .xlsx files for the subject
            for dirpath, _, filenames in os.walk(base_path_data + '/Joint_angles/' + subject):
                for filename in [f for f in filenames if f.endswith(".xlsx")]:
                    train_filenames.append(filename)
            print(filenames)

            # Process each discovered .xlsx file
            for filename in train_filenames:
                print(f"Loading data for file: {filename}")
                
                # Load AlphaPose data (.npy) for the file
                data = np.load(path_imu + subject + '/AlphaPose_' + filename[:-5] + '.npy', allow_pickle=True)

                # Load the corresponding labels
                label = np.loadtxt(base_path_data + '/Exercise_labels/'+ subject + '/' + filename[:-5] + '_labels.csv')

                # Mark and exclude double frames in labels
                # double_frames = np.load(base_path_data + '/Joint_angles/'+ subject + '/AlphaPose_' + filename[:-5] + '.npy')
                # for frame in double_frames:
                #     idx = int(frame[:-4])
                #     label[idx] = 99

                # Load pose data from Excel, excluding specified columns
                pose = pd.read_excel(
                    f"{base_path_data}/Joint_angles/{subject}/{filename}", 
                    sheet_name="Joint Angles XZY", 
                    usecols=lambda col: col not in self.skip_cols
                ).to_numpy()

                # Align data and label lengths
                min_len = min(len(data), len(label), len(pose))
                data, label, pose = data[:min_len], label[:min_len], pose[:min_len]

                # Filter out invalid frames (where label == 99)
                valid_indices = label != 99
                data, label, pose = data[valid_indices], label[valid_indices], pose[valid_indices]
                print(f" - Data, label, and pose shapes after filtering: {data.shape}, {label.shape}, {pose.shape}")

                if self.task == "videopose_test":
                    all_windows.append(data)
                    all_poses.append(pose)
                    all_poses = np.array(all_poses)
                    all_poses = all_poses.reshape(pose.shape[0], 17, 3)
                    all_poses = all_poses.tolist()
                    ground_truth = all_poses

                else:
                    # Generate sliding windows with the specified step size
                    middle_frame_indices = []
                    for start_idx in range(0, len(data) - self.window_len + 1, self.step_size):
                        end_idx = start_idx + self.window_len
                        if end_idx > len(data):
                            print(f"Skipping incomplete window: {start_idx} to {end_idx}")
                            break  # Skip incomplete windows at the end
                        
                        # Extract the window
                        window_data = data[start_idx:end_idx]
                        subject_windows.append(window_data)

                        # Capture the label or pose based on task
                        if self.task == "classification":
                            # Take label from the center of the window
                            window_label = label[start_idx + self.window_len // 2] # Take the central frame as ground truth label
                            # Map the label to its grouped exercise equivalent
                            window_label = map_labels_to_exercises([window_label])[0]
                            subject_labels.append(window_label)
                            ground_truth = subject_labels
                            #print(f'so far {len(subject_labels)} windows')
                        elif self.task == "getmb":
                            window_label = label[start_idx + self.window_len // 2]
                            subject_labels.append(window_label)
                            window_pose = pose[start_idx + self.window_len // 2].reshape(17, 3)  # Reshape for regression
                            subject_poses.append(window_pose)

                        elif self.task == "videopose_train":
                            window_pose = pose[start_idx + self.window_len // 2].reshape(17, 3)  # Reshape for regression
                            subject_poses.append(window_pose)
                            ground_truth = subject_poses
                    
                        else:
                            window_pose = pose[start_idx:end_idx].reshape(243, 17, 3)  # Reshape for regression
                            subject_poses.append(window_pose)
                            ground_truth = subject_poses


            # Append the data for this subject to all_windows
            if self.task == "getmb":
                all_windows.append((subject_windows, subject_labels, subject_poses))
            else:
                all_windows.append((subject_windows, ground_truth))

            # Print summary for each subject
            print(f"Subject {subject} - Total windows: {len(subject_windows)}")

        # Final summary for all subjects
        total_windows = sum(len(windows[0]) for windows in all_windows)
        print(f"Total samples across all subjects and files: {total_windows}")
        
        return all_windows

    def __len__(self):
        # Total windows across all subjects
        return sum(len(data[0]) for data in self.subject_data)

    def __getitem__(self, idx):
        # Find the correct subject and window within subject
        cum_size = 0
        for subject_data, ground_truth in self.subject_data:
            num_windows = len(subject_data)
            if idx < cum_size + num_windows:
                window_idx = idx - cum_size
                X = subject_data[window_idx]
                y = ground_truth[window_idx]

                if self.task == "classification":
                    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.int32)
                else:
                    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

            cum_size += num_windows


# DataLoader
def create_dataloader(subjects, window_len, step_size, test=False, task=None, batch_size=64, shuffle=True):
    dataset = APDataset(subjects=subjects, window_len=window_len, step_size=step_size, test=test, task=task)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True)
    return dataloader

def create_videopose_test_dataloader(subjects, test, task):
    dataset = FullSequenceDataset(subjects, test, task)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    return dataloader

