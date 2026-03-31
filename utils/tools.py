import numpy as np
import os, sys
import pickle
import yaml
from easydict import EasyDict as edict
from typing import Any, IO
from scipy.stats import mode
from collections import Counter

ROOT_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')

class TextLogger:
    def __init__(self, log_path):
        self.log_path = log_path
        with open(self.log_path, "w") as f:
            f.write("")
    def log(self, log):
        with open(self.log_path, "a+") as f:
            f.write(log + "\n")

class Loader(yaml.SafeLoader):
    """YAML Loader with `!include` constructor."""

    def __init__(self, stream: IO) -> None:
        """Initialise Loader."""

        try:
            self._root = os.path.split(stream.name)[0]
        except AttributeError:
            self._root = os.path.curdir

        super().__init__(stream)

def construct_include(loader: Loader, node: yaml.Node) -> Any:
    """Include file referenced at node."""

    filename = os.path.abspath(os.path.join(loader._root, loader.construct_scalar(node)))
    extension = os.path.splitext(filename)[1].lstrip('.')

    with open(filename, 'r') as f:
        if extension in ('yaml', 'yml'):
            return yaml.load(f, Loader)
        elif extension in ('json', ):
            return json.load(f)
        else:
            return ''.join(f.readlines())

def get_config(config_path):
    yaml.add_constructor('!include', construct_include, Loader)
    with open(config_path, 'r') as stream:
        config = yaml.load(stream, Loader=Loader)
    config = edict(config)
    _, config_filename = os.path.split(config_path)
    config_name, _ = os.path.splitext(config_filename)
    config.name = config_name
    return config

def ensure_dir(path):
    """
    create path by first checking its existence,
    :param paths: path
    :return:
    """
    if not os.path.exists(path):
        os.makedirs(path)
        
def read_pkl(data_url):
    file = open(data_url,'rb')
    content = pickle.load(file)
    file.close()
    return content

def smooth_predictions(predictions, smoothing_window=5, zero_replacement_window=30):
    """
    Smooth predictions by replacing isolated predictions with the majority label in the window.

    Parameters:
    - predictions (list or array): Sequence of predicted labels.
    - window_size (int): Size of the sliding window (should be an odd number).
    - threshold (float): Minimum fraction of the majority label required to replace a label.
    - max_iterations (int): Maximum number of iterations to apply smoothing.

    Returns:
    - smoothed_predictions (list): Smoothed predictions.
    """
    predictions = np.array(predictions)  # Ensure input is a NumPy array
    smoothed_predictions = predictions.copy()
    half_smoothing_window = smoothing_window // 2

    # Step 1: Apply majority label smoothing
    for i in range(len(predictions)):
        # Define the smoothing window range
        start = max(0, i - half_smoothing_window)
        end = min(len(predictions), i + half_smoothing_window + 1)
        window = smoothed_predictions[start:end]
        # Compute the majority label
        majority_label = mode(window).mode
        smoothed_predictions[i] = majority_label

    # Step 2: Replace `0` labels with the most frequent non-zero label in a larger window
    processed_predictions = smoothed_predictions.copy()
    half_zero_replacement_window = zero_replacement_window // 2

    for i in range(len(smoothed_predictions)):
        if smoothed_predictions[i] == 0:  # Only process `0` labels
            # Define the replacement window range
            start = max(0, i - half_zero_replacement_window)
            end = min(len(smoothed_predictions), i + half_zero_replacement_window + 1)
            window = smoothed_predictions[start:end]

            # Exclude zeros from the window
            non_zero_window = window[window != 0]

            if len(non_zero_window) > 4:  # Only replace if there are non-zero labels
                majority_label = mode(non_zero_window).mode
                processed_predictions[i] = majority_label

    return processed_predictions

def get_label_sequences(labels, target_label):
    """
    Finds the start and end indices of sequences where the label equals the target_label.

    Parameters:
        labels (list or array): List or array of labels.
        target_label (int): The target label to identify sequences for.

    Returns:
        list of tuples: Each tuple contains (start_index, end_index) for a sequence.
    """
    sequences = []
    start = None

    for i, label in enumerate(labels):
        if label == target_label and start is None:  # Start of a new sequence
            start = i
        elif label != target_label and start is not None:  # End of a sequence
            sequences.append((start, i - 1))
            start = None

    # Handle the case where a sequence ends at the last label
    if start is not None:
        sequences.append((start, len(labels) - 1))

    return sequences

################################## Exercise performance metrics tools #####################################
def compute_dur(labels):
    """
    Computes the total duration of segments in a sequence of labels.

    This function iterates through a sequence of segment labels (0–14), 
    identifying changes between segments and calculating their total duration. 
    The durations are normalized to minutes based on a sampling rate of 30 frames per second.

    Parameters:
        labels (list): A list of integers representing frame-by-frame segment labels.

    Returns:
        dict: A dictionary where keys are segment labels (0–14) and values are 
              the total duration of each segment in minutes, rounded to two decimal places.
"""
    segment_durations = {i: 0 for i in range(15)}  # Initialize dictionary for segment labels 0 to 14
    current_segment = None
    current_duration = 0

    for label in labels:
        if label != current_segment:
            # If we encounter a new segment and there's an existing segment, update its duration
            if current_segment is not None:
                segment_durations[current_segment] += current_duration

            # Start tracking the new segment
            current_segment = label
            current_duration = 1  # Increment duration for the new segment
        else:
            # Continue tracking the same segment
            current_duration += 1

    # Append the duration for the last segment
    if current_segment is not None:
        segment_durations[current_segment] += current_duration

    segment_durations = {segment: round(duration / (30 * 60), 2) for segment, duration in segment_durations.items()}

    return segment_durations

def compute_rep_count(labels):
    """
    Counts the number of repetitions for each exercise segment using majority voting 
    and tracks the start and end indices of each segment.

    Parameters:
        labels (list): A list containing frame-by-frame labels for a CV fold.
    Returns:
        tuple: 
            rep_count (dict): A dictionary where keys are exercise labels and values 
                              are the count of repetitions.
            rep_ids (dict): A dictionary where keys are exercise labels and 
                                 values are lists of (start, end) indices for each segment.
    """
    rep_count = {}
    rep_ids = {}
    window_size = 30 # Size of the window used for majority voting.
    current_segment = None
    start_idx = None

    for idx in range(len(labels)):  # Iterate through each label
        # Determine the majority label for the current window
        start_window = max(0, idx - window_size // 2)
        end_window = min(len(labels), idx + window_size // 2 + 1)
        majority_label = Counter(labels[start_window:end_window]).most_common(1)[0][0]

        # Detect segment changes based on majority voting
        if majority_label != current_segment:
            if current_segment is not None:  # Save the previous segment
                if current_segment not in rep_count:
                    rep_count[current_segment] = 0
                    rep_ids[current_segment] = []
                rep_count[current_segment] += 1
                rep_ids[current_segment].append((start_idx, idx - 1))

            # Update to the new segment
            current_segment = majority_label
            start_idx = idx

    # Handle the last segment
    if current_segment is not None:
        if current_segment not in rep_count:
            rep_count[current_segment] = 0
            rep_ids[current_segment] = []
        rep_count[current_segment] += 1
        rep_ids[current_segment].append((start_idx, len(labels) - 1))

    return rep_count, rep_ids

def compute_mad(joint_data, segment_indices, classes):
    """
    Computes the motion variability in terms of the Mean Absolute Deviation (MAD) 
    for the predicted values using provided segment start and end indices.

    Parameters:
        joint_data (np.ndarray): Predicted axial data, a 4D array with shape 
                                 (sequences, frames, joints, coordinates).
        segment_indices (dict): A dictionary where keys are class labels and values 
                                are lists of (start, end) tuples representing segment indices.
        classes (dict): Dictionary where keys are class labels and values 
                        represent the number of segments per class.

    Returns:
        dict: A dictionary where keys are class labels and values 
              are lists of MADs for each segment of the respective class.
    """
    # Flatten joint data for easy indexing
    joint_data_flat = joint_data.reshape(-1, joint_data.shape[2], joint_data.shape[3])

    # Extract the x-coordinate (Lateral Bending) for the specified joint (joint ID 11)
    joint_x = joint_data_flat[:, 11, 2]

    # Dictionary to store MADs for each class
    mad_per_class = {label: [] for label in classes.keys()}

    # Iterate over specified class labels
    for class_label, num_segments in classes.items():
        # Retrieve precomputed segment indices for the current class
        if class_label not in segment_indices:
            continue

        label_seq = segment_indices[class_label]

        # Compute MAD for each sequence (limit to the specified number of segments)
        for i, (start, end) in enumerate(label_seq):
            if i >= num_segments:  # Stop after processing the specified number of segments
                break

            # Extract x-coordinates for the sequence
            pred_segment = joint_x[start:end + 1]

            # Compute Mean Absolute Deviation (MAD) for the predicted segment
            if len(pred_segment) > 0:
                pred_mad = round(np.mean(np.abs(pred_segment - np.mean(pred_segment))), 0)
                mad_per_class[class_label].append(pred_mad)
            else:
                mad_per_class[class_label].append(np.nan)

    return mad_per_class

