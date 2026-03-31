import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MLP(nn.Module):
    '''
        Single layer MLP for either 3D joint angle prediction (regression) or action classification.
        Input: (N, M, T, J, C)
    '''
    def __init__(self, task='regression', num_classes=15, dropout_ratio=0.5, dim_rep=512, hidden_dim=2048, window_len=243, num_joints=17, output_dim=3, mb_representation=False):
        super(MLP, self).__init__()
        #self.backbone = backbone
        self.task = task  # 'regression' or 'classification'
        self.mb_representation = mb_representation
        self.dropout = nn.Dropout(p=dropout_ratio)
        self.hidden_dim = hidden_dim
        self.num_joints = num_joints
        self.output_dim = output_dim
        self.window_len = window_len
        self.bn = nn.BatchNorm1d(hidden_dim, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(dim_rep, hidden_dim)
        
        # Define output layer based on the task
        if self.task == 'regression':
            self.fc2 = nn.Linear(hidden_dim, num_joints * output_dim)
        elif self.task == 'classification':
            self.fc2 = nn.Linear(hidden_dim, num_classes)
            
    def forward(self, feat):
        feat = self.dropout(feat)
        if self.mb_representation:
            N, J, C = feat.shape
        else:
            # with torch.no_grad():
            #     feat = self.backbone.module.get_representation(feat)
            N, T, J, C = feat.shape  # Unpacking the dimensions
            feat = feat.mean(dim=1)
        
        feat = feat.view(N, J * C)  # flatten the array to a 2D feature vector (n_batches, 17*3)
        
        feat = self.fc1(feat)
        feat = self.bn(feat)
        feat = self.relu(feat)
        feat = self.fc2(feat)
        
        # Reshape output for regression task
        if self.task == 'regression':
            feat = feat.view(-1, self.num_joints, self.output_dim)  # (batch_size, num_joints, output_dim)
        
        return feat