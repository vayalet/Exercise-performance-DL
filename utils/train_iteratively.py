from matplotlib import pyplot as plt
import numpy as np
import os
import time
import sys
import argparse
import errno
from collections import OrderedDict, defaultdict, Counter
import tensorboardX
from tqdm import tqdm
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader, TensorDataset, ConcatDataset

from data.loader import *
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from einops import rearrange
from utils.tools import *
from utils.learning import *
from models import mlp, simple, videopose, tcn
import glob
import h5py


random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

def initialize_model(model_version, **kwargs):
    if model_version == 'mlp':
        return mlp.MLP(**kwargs)
    elif model_version == 'simple':
        required_args = {k: kwargs[k] for k in [
            'task', 'num_classes', 'dropout_ratio', 'dim_rep', 
            'window_len', 'num_joints', 'output_dim', 'mb_representation'
        ] if k in kwargs}
        return simple.Simple(**required_args)
    elif model_version == 'videopose':
        required_args = {k: kwargs[k] for k in [
            'task', 'num_classes', 'dropout_ratio', 'num_joints'
        ] if k in kwargs}
        architecture = '3,3,3,3,3'
        filter_widths = [int(x) for x in architecture.split(',')]
        return videopose.TemporalModelOptimized1f(required_args.get('num_joints'), 2, required_args.get('num_classes'),
                                        filter_widths=filter_widths, causal=False, dropout=required_args.get('dropout_ratio'), channels=1024, task=required_args.get('task'))
    elif model_version == 'tcn':
        required_args = {k: kwargs[k] for k in [
            'task', 'num_classes', 'dropout_ratio', 'num_joints'
        ] if k in kwargs}
        architecture = '3,3,3,3,3'
        filter_widths = [int(x) for x in architecture.split(',')]
        return tcn.SimpleTemporalModel(required_args.get('num_joints'), 3, required_args.get('num_classes'),
                                        filter_widths=filter_widths, dropout=required_args.get('dropout_ratio'), channels=1024, task=required_args.get('task'))
    else:
        raise ValueError("Unknown model version specified.")

def train(loss_weights, args, opts, train_loader, model_version, early_stopping_patience=5, **model_kwargs):
    model = initialize_model(model_version=model_version, **model_kwargs)
    if(args.mb_representation == True):
        opts.checkpoint = 'checkpoint/class/mb_backbone'
    else: 
        opts.checkpoint = 'checkpoint/class/scratch'

    train_writer = tensorboardX.SummaryWriter(os.path.join(opts.checkpoint, "logs"))
    logs_folder = os.path.join(opts.checkpoint, "logs")
    logs_path = os.path.join(logs_folder, f'{opts.model_version}_metrics_{opts.split}.h5')

    criterion = nn.CrossEntropyLoss(weight=loss_weights) if opts.task == 'classification' else nn.L1Loss(reduction='mean')
    if torch.cuda.is_available():
        model = nn.DataParallel(model)
        model = model.cuda()
        criterion = criterion.cuda()
        loss_weights = loss_weights.cuda()

    optimizer = optim.Adam(
        model.parameters(),     
        lr=args.learning_rate,              
        betas=(0.9, 0.999),     
        eps=1e-8,              
        weight_decay=args.weight_decay      
    )
    scheduler = StepLR(optimizer, step_size=1, gamma=args.lr_decay)

    best_f1 = 0
    epochs_no_improve = 0  # Early stopping counter
    with h5py.File(logs_path, 'w') as f:
        f.create_dataset('train_loss', shape=(args.epochs,), dtype=np.float32)
        f.create_dataset('train_acc', shape=(args.epochs,), dtype=np.float32)
        f.create_dataset('train_f1', shape=(args.epochs,), dtype=np.float32)

    for epoch in range(args.epochs):
        model.train()
        losses_train = AverageMeter()
        f1 = AverageMeter()
        acc = AverageMeter()
        train_loader = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for idx, (batch_input, batch_gt) in enumerate(train_loader):
            if opts.model_version == "videopose":
                batch_input = batch_input[..., :2]
            batch_input, batch_gt = batch_input.float(), batch_gt.long()
            if torch.cuda.is_available():
                batch_gt = batch_gt.cuda()
                batch_input = batch_input.cuda()
            output = model(batch_input)
            optimizer.zero_grad()
            loss_train = criterion(output, batch_gt)

            losses_train.update(loss_train.item(), batch_input.size(0))
            output = output.argmax(dim=1).cpu().detach().numpy()
            batch_gt = batch_gt.cpu().detach().numpy()
            acc.update(accuracy_score(batch_gt, output), batch_input.size(0))
            f1.update(f1_score(batch_gt, output, average='weighted'), batch_input.size(0))
            loss_train.backward()
            optimizer.step()    

        train_writer.add_scalar('train_loss', losses_train.avg, epoch + 1)
        train_writer.add_scalar('train_acc', acc.avg, epoch + 1)
        train_writer.add_scalar('train_f1', f1.avg, epoch + 1)

        scheduler.step()

        # Early Stopping Check
        if f1.avg > best_f1:
            best_f1 = f1.avg
            epochs_no_improve = 0
            best_chk_path = os.path.join(opts.checkpoint, f'{model_version}_best_epoch_split{opts.split}.bin')
            torch.save(model.state_dict(), best_chk_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stopping_patience:
                print("Early stopping triggered.")
                break

        with h5py.File(logs_path, 'a') as f:
            f['train_loss'][epoch] = losses_train.avg
            f['train_acc'][epoch] = acc.avg
            f['train_f1'][epoch] = f1.avg
        print(f'Epoch {epoch+1}: Train Acc: {acc.avg}, F1: {f1.avg}')

    return model


def evaluate_model(test_loader, model, opts, args):
    print("Evaluating the model on test data")
    model.eval()
    if torch.cuda.is_available():
        model = nn.DataParallel(model) if not isinstance(model, nn.DataParallel) else model
        model = model.cuda()

    chk_filename = os.path.join(opts.checkpoint, f'{opts.model_version}_best_epoch_split{opts.split}.bin')
    checkpoint = torch.load(chk_filename)
    model.load_state_dict(checkpoint, strict=True)

    all_preds = []
    all_labels = []
    test_loader = tqdm(test_loader, desc="Testing")
    with torch.no_grad():
        for batch_input, batch_gt in test_loader:
            if isinstance(model.module, videopose.TemporalModel):
                batch_input = batch_input[..., :2]
            batch_input, batch_gt = batch_input.float(), batch_gt.long()
            if torch.cuda.is_available():
                batch_input = batch_input.cuda()
                batch_gt = batch_gt.cuda()
            predictions = model(batch_input)
            all_preds.extend(predictions.argmax(dim=1).cpu().numpy())
            all_labels.extend(batch_gt.cpu().numpy())

    # Calculate metrics and save
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    results = {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}

    result_path = f'results/class/subset_data{("mb_backbone" if args.mb_representation else "scratch")}/{opts.model_version}_{"MB" if args.mb_representation else "scratch"}_split{opts.split}_metrics.npy'
    np.save(result_path, results)
    print("Test results saved:", results)