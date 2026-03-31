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
        base_path_data = '/scratch/leuven/365/vsc36565/motionbert/data/test/' if self.test else '/scratch/leuven/365/vsc36565/ai@wzc/data/'
        path_imu = '/scratch/leuven/365/vsc36565/motionbert/results_test_pose/test/' if self.test else '/scratch/leuven/365/vsc36565/ai@wzc/data/results_test_pose/'

        all_data = []
        for subject in self.subjects:
            print(f"Processing subject {subject}")
            subject_data = []
            subject_poses = []
            subject_labels = []
            train_filenames = []

            # Discover .xlsx files for the subject
            for dirpath, _, filenames in os.walk(base_path_data + subject):
                for filename in [f for f in filenames if f.endswith(".xlsx")]:
                    train_filenames.append(filename)

            # Process each discovered .xlsx file
            for filename in train_filenames:
                print(f"Loading data for file: {filename}")
                
                # Load AlphaPose data (.npy) for the file
                data = np.load(path_imu + subject + '/AlphaPose_' + filename[:-5] + '.npy', allow_pickle=True)

                # Load the corresponding labels
                label = np.loadtxt(base_path_data + subject + '/' + filename[:-5] + '_d-el.csv')

                # Mark and exclude double frames in labels
                double_frames = np.load(base_path_data + subject + '/AlphaPose_' + filename[:-5] + '.npy')
                for frame in double_frames:
                    idx = int(frame[:-4])
                    label[idx] = 99

                # Load pose data from Excel, excluding specified columns
                pose = pd.read_excel(
                    f"{base_path_data}{subject}/{filename}", 
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
        base_path_data = '/scratch/leuven/365/vsc36565/motionbert/data/test/' if self.test else '/scratch/leuven/365/vsc36565/ai@wzc/data/'
        path_imu = '/scratch/leuven/365/vsc36565/motionbert/results_test_pose/test/' if self.test else '/scratch/leuven/365/vsc36565/ai@wzc/data/results_test_pose/'

        all_poses = []
        all_windows = []
        for subject in self.subjects:
            print(f"Processing subject {subject}")
            subject_windows = []
            subject_labels = []
            subject_poses = []
            train_filenames = []

            # Discover .xlsx files for the subject
            for dirpath, _, filenames in os.walk(base_path_data + subject):
                for filename in [f for f in filenames if f.endswith(".xlsx")]:
                    train_filenames.append(filename)
            print(filenames)

            # Process each discovered .xlsx file
            for filename in train_filenames:
                print(f"Loading data for file: {filename}")
                
                # Load AlphaPose data (.npy) for the file
                data = np.load(path_imu + subject + '/AlphaPose_' + filename[:-5] + '.npy', allow_pickle=True)

                # Load the corresponding labels
                label = np.loadtxt(base_path_data + subject + '/' + filename[:-5] + '_d-el.csv')

                # Mark and exclude double frames in labels
                double_frames = np.load(base_path_data + subject + '/AlphaPose_' + filename[:-5] + '.npy')
                for frame in double_frames:
                    idx = int(frame[:-4])
                    label[idx] = 99

                # Load pose data from Excel, excluding specified columns
                pose = pd.read_excel(
                    f"{base_path_data}{subject}/{filename}", 
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


# Data loader for limited data scenario
def create_limited_dataloader(subjects, window_len, step_size, filter_label, filter_percentage, test=False, task=None, batch_size=64, shuffle=True):
    dataset = APDatasetFiltered(subjects, window_len, step_size, test=False, task=task, filter_label=filter_label, filter_percentage=filter_percentage)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True)
    return dataloader

def load_mb_repr(subjects, base_path, task, batch_size, window_len, shuffle):
    # mb_data = {}
    # for s_id, subject in enumerate(subjects):
    #     print('MB data/', subject)
    #     #path_data = f"/scratch/leuven/365/vsc36565/motionbert/results_mb_repr/{subject}/representations_{subject}.h5"
    #     path_data = f"/scratch/leuven/365/vsc36565/motionbert/results_mb_repr/{subject}/representations_test.h5"
    #     with h5py.File(path_data, 'r') as f:
    #     # Load the dataset from the HDF5 file
    #         data = f['representations'][:]
    #     X = data
    #     mb_data[subject] = X

    #dataset = MBDataset(subjects, base_path, window_len)
    dataset = MBRepresentationDataset(subjects, base_path, task)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=1, pin_memory=True)
    print("MB data is loaded")
    return dataloader

class MBRepresentationDataset(Dataset):
    def __init__(self, subjects, base_path, task):
        """
        Initialize the dataset with multiple subjects, loading data into memory.

        Parameters:
        - subjects: List of subject identifiers (e.g., ['subject1', 'subject2'])
        - base_path: Base path where each subject's HDF5 file is stored
        - task: Task type, either "classification" or "regression"
        """
        self.subjects = subjects
        self.base_path = base_path
        self.task = task
        self.representations = []
        self.ground_truths = []

        self._load_all_data()  # Load data into memory

    def _load_all_data(self):
        """
        Load all representations and ground truths into memory for faster access.
        """
        for subject in self.subjects:
            file_path = os.path.join(self.base_path, f"{subject}/mb_data_243win.h5")
            with h5py.File(file_path, 'r') as f:
                # Load representations and labels/poses into memory
                self.representations.append(torch.tensor(f['representations'][:], dtype=torch.float32))
                if self.task == "classification":
                    self.ground_truths.append(torch.tensor(f['labels'][:], dtype=torch.long))
                else:
                    self.ground_truths.append(torch.tensor(f['poses'][:], dtype=torch.float32))

        # Concatenate all subject data
        self.representations = torch.cat(self.representations, dim=0)
        self.ground_truths = torch.cat(self.ground_truths, dim=0)
        print(f"Total data loaded: {len(self.representations)} samples.")

    def __len__(self):
        return len(self.representations)

    def __getitem__(self, idx):
        """
        Retrieve the representation and corresponding ground truth from memory.
        """
        return self.representations[idx], self.ground_truths[idx]


class MBDataset(Dataset):
    def __init__(self, subjects, base_path, window_len):
        self.subjects = subjects
        self.base_path = base_path
        self.window_len = window_len

        # Load entire data for each subject into memory
        self.data, self.labels, self.cum_sizes = self._load_all_subjects_into_memory()

    def _load_all_subjects_into_memory(self):
        """Load raw data for each subject and keep cumulative sizes for indexing."""
        data_list = []
        label_list = []
        cum_sizes = []

        for subject in self.subjects:
            print(f'Loading data for {subject}')
            file_path = os.path.join(self.base_path, f"{subject}/representations_243win.h5")
            label_file_path = os.path.join(self.base_path, f"{subject}/labels_243win.h5")
            
            with h5py.File(file_path, 'r') as rep_file:
                subject_data = torch.tensor(rep_file['representations'][:], dtype=torch.float32)

            with h5py.File(label_file_path, 'r') as label_file:
                subject_labels = torch.tensor(label_file['labels'][:], dtype=torch.int64)

            # Add data and labels
            data_list.append(subject_data)
            label_list.append(subject_labels)
            
            # Track cumulative size to map global index to subject index
            if cum_sizes:
                cum_sizes.append(cum_sizes[-1] + len(subject_data) - self.window_len + 1)
            else:
                cum_sizes.append(len(subject_data) - self.window_len + 1)

            print(f'Data shape {subject_data.shape}, labels shape {subject_labels.shape}')

        return data_list, label_list, cum_sizes

    def __len__(self):
        # Total number of windows across all subjects
        return self.cum_sizes[-1]

    def __getitem__(self, idx):
        # Find which subject this index belongs to
        for subj_idx, cum_size in enumerate(self.cum_sizes):
            if idx < cum_size:
                if subj_idx > 0:
                    idx -= self.cum_sizes[subj_idx - 1]
                break

        # Get the subject's data and generate the window
        subject_data = self.data[subj_idx]
        subject_labels = self.labels[subj_idx]

        # Extract the window starting at the specified index
        # window_data = subject_data[idx:idx + self.window_len]
        # label = subject_labels[idx + self.window_len // 2]  # Central label for the window
        data_point = subject_data[idx]
        label = subject_labels[idx]

        #return window_data, label
        return data_point, label


