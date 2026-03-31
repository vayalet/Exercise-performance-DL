import numpy as np
import os
import time
import sys
import argparse
import errno
from collections import OrderedDict, defaultdict, Counter 
from tqdm import tqdm
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset, ConcatDataset

from utils.loader import *
from sklearn.metrics import accuracy_score, f1_score
from utils.tools import *
from utils.learning import *
from models import videopose

import h5py

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/train_benchmarks.yaml", help="Path to the config file.")
    parser.add_argument('-c', '--checkpoint', default='./results/class/videopose', type=str, metavar='PATH', help='checkpoint directory')
    parser.add_argument('-s', '--test_subject', type=str, default='sub01', help='Test subject for evaluation')
    parser.add_argument('--split', type=int, default=0, help='Split ID for LOSOCV')
    parser.add_argument('--task', type=str, default='classification', help='Classification or regression')
    parser.add_argument('--pretrained', default='./results/class/latest_epoch.bin', type=str, metavar='FILENAME', help='checkpoint for pretrained model')
    parser.add_argument('--experiment', type=str, default='videopose', help='Model selection (only videopose is supported)')
    opts = parser.parse_args()
    return opts


random.seed(0)
np.random.seed(0)
torch.manual_seed(0)


def train(train_loader, args, opts, class_weights):
    """Train the VideoPose model for either exercise classification or pose regression."""

    architecture = '3,3,3,3,3'
    filter_widths = [int(x) for x in architecture.split(',')]

    # The output head differs by task:
    # - classification predicts exercise labels
    # - regression predicts 3D joint angles
    if opts.task == "regression":
        opts.checkpoint = './results/reg'
        model = videopose.TemporalModelOptimized1f(
            17, 3, 17,
            filter_widths=filter_widths, causal=False, dropout=args.dropout, channels=1024, task=opts.task
        )
    else:
        opts.checkpoint = './results/class'
        model = videopose.TemporalModelOptimized1f(
            17, 3, 15,  # or 15 for all classes
            filter_widths=filter_widths, causal=False, dropout=args.dropout, channels=1024, task=opts.task
        )

    logs_folder = os.path.join(opts.checkpoint, "logs")
    logs_path = os.path.join(logs_folder, f'metrics_{opts.split}.h5')

    criterion = nn.CrossEntropyLoss(weight=class_weights) if opts.task == 'classification' else nn.MSELoss(reduction='mean')
    if torch.cuda.is_available():
        print("CUDA is available! Using GPU.")
        model = nn.DataParallel(model).cuda()
        criterion = criterion.cuda()
    else: 
        print("CUDA is not available! Using CPU.")

    model_params = 0
    for parameter in model.parameters():
        if parameter.requires_grad:  # Only count trainable parameters
            model_params += parameter.numel()
    print('INFO: Trainable parameter count:', model_params)
        
    optimizer = optim.Adam(model.parameters(), lr=args.lr, amsgrad=True)

    lr_decay = args.lr_decay
    lr = args.lr

    initial_momentum = 0.1
    final_momentum = 0.001
    
    print('INFO: Training on {} batches'.format(len(train_loader)))
    
    with h5py.File(logs_path, 'w') as f:
        f.create_dataset('train_loss', shape=(args.epochs,), dtype=np.float32)
        
        if opts.task == "classification":
            f.create_dataset('train_acc', shape=(args.epochs,), dtype=np.float32)
            f.create_dataset('train_f1', shape=(args.epochs,), dtype=np.float32)
        elif opts.task == "regression":
            f.create_dataset('train_mae', shape=(args.epochs,), dtype=np.float32)
            f.create_dataset('test_mae', shape=(args.epochs,), dtype=np.float32)
    
    # Training loop
    for epoch in range(args.epochs):
        print(f'Training epoch {epoch + 1}.')
        total_loss = 0
        
        if opts.task == "classification":
            all_preds = []
            all_targets = []
        else:  # Initialize for regression task
            total_mae = 0
            num_samples = 0

        model.train()
        train_loader = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for idx, (batch_input, batch_gt) in enumerate(train_loader):

            train_loader.set_postfix({"Batch": idx})
            
            # Move data to CUDA if available
            batch_input = batch_input.float().cuda() if torch.cuda.is_available() else batch_input.float()
            batch_gt = batch_gt.cuda() if torch.cuda.is_available() else batch_gt
            batch_gt = batch_gt.unsqueeze(1)

            # Set correct data type for targets based on task
            if opts.task == "classification":
                batch_gt = batch_gt.long()
            else:
                batch_gt = batch_gt.float()

            batch_size = len(batch_input)

            # Forward pass
            optimizer.zero_grad()

            output = model(batch_input)
            if opts.task == "classification":
                output = output.squeeze(1)
                batch_gt = batch_gt.squeeze()
                loss_train = criterion(output, batch_gt)
            else:
                # Only a subset of joints is weighted in the MAE loss.
                joint_mask = torch.tensor([1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1], dtype=torch.float32)
                loss_train = mae_loss(output, batch_gt, joint_mask)

            # Backward pass
            loss_train.backward()
    
            optimizer.step()
        
            # Accumulate loss
            total_loss += loss_train.item()

            # Additional calculations based on task
            if opts.task == "classification":
                # Calculate predictions and accumulate for accuracy and F1 calculation
                preds = output.argmax(dim=1).cpu().detach().numpy()
                all_preds.extend(preds)
                all_targets.extend(batch_gt.cpu().detach().numpy())
            elif opts.task == "regression":
                batch_mae = mae(output, batch_gt, joint_mask)
                total_mae += batch_mae * batch_size  # Accumulate MAE weighted by batch size
                num_samples += batch_size
        
        # Decay learning rate exponentially
        lr *= lr_decay
        print(f'Learning rate: {lr}')
        for param_group in optimizer.param_groups:
            param_group['lr'] *= lr_decay

        # # Decay BatchNorm momentum
        # momentum = initial_momentum * np.exp(-(epoch+1)/args.epochs * np.log(initial_momentum/final_momentum))
        # model.module.set_bn_momentum(momentum)


        # Calculate metrics for the training set
        avg_loss = total_loss / len(train_loader)
        
        if opts.task == "classification":
            train_acc = accuracy_score(all_targets, all_preds)
            train_f1 = f1_score(all_targets, all_preds, average='weighted')
        elif opts.task == "regression":
            avg_mae = total_mae / num_samples  # Compute mean MAE across all samples

        # Save metrics to file
        with h5py.File(logs_path, 'a') as f:
            f['train_loss'][epoch] = avg_loss

            if opts.task == "classification":
                f['train_acc'][epoch] = train_acc
                f['train_f1'][epoch] = train_f1
            elif opts.task == "regression":
                f['train_mae'][epoch] = avg_mae  # Mean Angular Error for training
                
        # Print relevant metrics
        print(f'Epoch {epoch+1} - Train Loss: {avg_loss:.4f}')
        
        if opts.task == "classification":
            print(f'Epoch {epoch+1} - Train Accuracy: {train_acc:.4f}, Train F1: {train_f1:.4f}')
        elif opts.task == "regression":
            print(f'Epoch {epoch+1} - Train MAE: {avg_mae:.4f}')

        # Save latest checkpoint 
        chk_path = os.path.join(opts.checkpoint, f'latest_epoch_split{opts.split}.bin')
        print(f'Saving latest checkpoint to {chk_path}')
        torch.save({
            'epoch': epoch + 1,
            'lr': args.lr,
            'optimizer': optimizer.state_dict(),
            'model': model.state_dict(),
            'f1': train_f1 if opts.task == "classification" else None,
            'mae': avg_mae if opts.task == "regression" else None
        }, chk_path)
    


def evaluate_model(test_loader, opts, args):
    """Load the latest checkpoint and save predictions on the test subject."""
    print('Evaluating the videopose model on test data')

    # Use the same temporal architecture as in training, but with the full inference model.
    architecture = '3,3,3,3,3'
    filter_widths = [int(x) for x in architecture.split(',')]
    if opts.task == "regression":
        opts.checkpoint = './results/reg'
        model = videopose.TemporalModel(
            17, 3, 17,
            filter_widths=filter_widths, causal=False, dropout=args.dropout, channels=1024, task=opts.task
        )
    else:
        opts.checkpoint = './results/class'
        model = videopose.TemporalModel(
            17, 3, 15,  # or 15 for all classes
            filter_widths=filter_widths, causal=False, dropout=args.dropout, channels=1024, task=opts.task
        )
        
    # Load the checkpoint for the entire model
    chk_path = os.path.join(opts.checkpoint, f'latest_epoch_split{opts.split}.bin')
    print(f'Loading latest checkpoint from {chk_path}')
    checkpoint = torch.load(chk_path, map_location=lambda storage, loc: storage)

    # Confirm that the model is in evaluation mode
    model.eval()
    
    if torch.cuda.is_available():
        model = nn.DataParallel(model) if not isinstance(model, nn.DataParallel) else model
        model = model.cuda()

    # Load the weights into the full model
    model.load_state_dict(checkpoint['model'], strict=True)

    # Set up result paths based on the task
    if opts.task == "classification":
        result_path = f'./results/class/videopose_split{opts.split}.npy'
        window_preds = torch.zeros(0, dtype=torch.long, device='cpu')
        true_labels = []
    elif opts.task == "regression":
        joint_mask = torch.tensor([1,1,1,0,0,0,0,0,0,0,0,1,1,1,1,1,1], dtype=torch.float32)
        joint_mask = joint_mask.view(1, -1, 1).cpu()
        result_path = f'./results/reg/videopose_split{opts.split}.npy'
        

    test_loader = tqdm(test_loader, desc="Testing")

    with torch.no_grad():
        for id, (batch_input, batch_gt) in enumerate(test_loader):
            test_loader.set_postfix({"Batch": id})
            batch_size = len(batch_input)
            batch_input = batch_input.float()
            if opts.task == "classification":
                batch_gt = batch_gt.long()
            else:
                batch_gt = batch_gt.float()
            
            # Move inputs to GPU if available
            if torch.cuda.is_available():
                batch_input = batch_input.cuda()
                batch_gt = batch_gt.cuda()

            # Forward pass
            predictions = model(batch_input) # (batch_size, 1, 15) for class

            # Classification processing
            if opts.task == "classification":
                predictions = predictions.squeeze(1)
                predicted = predictions.argmax(dim=1).cpu()
                window_preds = torch.cat([window_preds, predicted.view(-1)])
                true_labels.extend(batch_gt.cpu().numpy().tolist())

            # Regression processing with incremental saving
            elif opts.task == "regression":
                predictions = predictions.cpu().squeeze(0) * joint_mask
                true_batch_pose = batch_gt.cpu().squeeze(0) * joint_mask

                # Convert predictions and true poses to numpy
                predictions_np = predictions.cpu().numpy()
                true_batch_pose_np = true_batch_pose.cpu().numpy()

                # Concatenate predictions and true poses along the joint dimension
                batch_results = np.concatenate((predictions_np, true_batch_pose_np), axis=1)
                if id == 0:
                    all_results = batch_results
                else:
                    all_results = np.concatenate((all_results, batch_results), axis=0)


    # Final save for classification
    if opts.task == "classification":
        frame_predictions = window_preds.numpy().tolist()
        results = np.stack((frame_predictions, true_labels), axis=1)
        np.save(result_path, results)
        print("Classification test results have been saved!")
    else:
        np.save(result_path, all_results)
        print("Regression test results have been saved!")

def calculate_class_weights(dataloader):
    """
    Calculate class weights based on the frequency of each class in the training dataset.
    """
    label_counts = Counter({i: 0 for i in range(15)})
    # label_counts = Counter({i: 0 for i in range(7)}) # for the combined exercises
    
    for _, labels in dataloader:
        label_counts.update(labels.tolist())  # Convert tensor to list and update count
    
    # Calculate the total number of labels
    total_labels = sum(label_counts.values())
    
    # Convert counts to a tensor, ensuring every class from 0 to 14 is included or to 6 for the combined exercises
    counts_tensor = torch.tensor([label_counts[i] for i in range(15)], dtype=torch.float32)
    
    # Avoid zero division by adding a small constant
    counts_tensor = counts_tensor + 1e-6
    
    # Calculate weights inversely proportional to class frequencies
    weights_tensor = total_labels / (15 * counts_tensor)
    
    return weights_tensor

def load_data(subjects):
    """Create the training and test loaders for a leave-one-subject-out split."""
    print(f'Split: {opts.split}, test subject: {opts.test_subject}')
    remaining_subjects = [subject for subject in subjects if subject != opts.test_subject]
    train_subjects = [subject for subject in subjects if subject != opts.test_subject]
    print(f'Train subjects: {train_subjects}')

    test = False
    if opts.task == "regression":
        # Regression trains on windowed clips but evaluates on the full padded sequence.
        train_loader = create_dataloader(train_subjects, args.window_len, args.step_size, test=test, task="videopose_train", batch_size=args.batch_size, shuffle=True)
        test_loader = create_videopose_test_dataloader([opts.test_subject], test=test, task=opts.task)
    else:
        # Classification uses sliding windows for both training and evaluation.
        train_loader = create_dataloader(train_subjects, args.window_len, args.step_size, test=test, task=opts.task, batch_size=args.batch_size, shuffle=True)
        test_loader = create_dataloader([opts.test_subject], 243, 1, test=test, task=opts.task, batch_size=args.batch_size, shuffle=False)

    return train_loader, test_loader

if __name__ == "__main__":
    # Make paths relative to this script and load the YAML config.
    os.chdir(sys.path[0])
    opts = parse_args()
    args = get_config(opts.config)
    print(args)

    # These hard-coded values act as the default local experiment setup.
    # opts.split = 0
    # opts.experiment = 'videopose'
    # opts.test_subject = 'sub01'
    # opts.task = 'regression'

    subjects = ['sub01', 'sub02', 'sub03', 'sub04', 'sub05', 'sub06', 'sub07']
    train_loader, test_loader = load_data(subjects)
    if opts.task == "classification":
        if train_loader != None:
            class_weights = calculate_class_weights(train_loader)
            print(class_weights)
    else:
        class_weights = None
    train(train_loader, args, opts, class_weights)
    evaluate_model(test_loader, opts, args)
