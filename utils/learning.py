import os
import numpy as np
import torch
import torch.nn as nn
from functools import partial
from models.DSTformer import DSTformer

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def load_pretrained_weights(model, checkpoint):
    """Load pretrianed weights to model
    Incompatible layers (unmatched in name or size) will be ignored
    Args:
    - model (nn.Module): network model, which must not be nn.DataParallel
    - weight_path (str): path to pretrained weights
    """
    import collections
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    model_dict = model.state_dict()
    new_state_dict = collections.OrderedDict()
    matched_layers, discarded_layers = [], []
    for k, v in state_dict.items():
        # If the pretrained state_dict was saved as nn.DataParallel,
        # keys would contain "module.", which should be ignored.
        if k.startswith('module.'):
            k = k[7:]
        if k in model_dict and model_dict[k].size() == v.size():
            new_state_dict[k] = v
            matched_layers.append(k)
        else:
            discarded_layers.append(k)
    model_dict.update(new_state_dict)
    model.load_state_dict(model_dict, strict=True)
    print('load_weight', len(matched_layers))
    return model

def partial_train_layers(model, partial_list):
    """Train partial layers of a given model."""
    for name, p in model.named_parameters():
        p.requires_grad = False
        for trainable in partial_list:
            if trainable in name:
                p.requires_grad = True
                break
    return model

def load_backbone(args):
    if not(hasattr(args, "backbone")):
        args.backbone = 'DSTformer' # Default
    if args.backbone=='DSTformer':
        model_backbone = DSTformer(dim_in=3, dim_out=3, dim_feat=args.dim_feat, dim_rep=args.dim_rep, 
                                   depth=args.depth, num_heads=args.num_heads, mlp_ratio=args.mlp_ratio, norm_layer=partial(nn.LayerNorm, eps=1e-6), 
                                   maxlen=args.maxlen, num_joints=args.num_joints)
    elif args.backbone=='TCN':
        from lib.model.model_tcn import PoseTCN
        model_backbone = PoseTCN()
    elif args.backbone=='poseformer':
        from lib.model.model_poseformer import PoseTransformer 
        model_backbone = PoseTransformer(num_frame=args.maxlen, num_joints=args.num_joints, in_chans=3, embed_dim_ratio=32, depth=4,
            num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,drop_path_rate=0, attn_mask=None) 
    elif args.backbone=='mixste':
        from lib.model.model_mixste import MixSTE2 
        model_backbone = MixSTE2(num_frame=args.maxlen, num_joints=args.num_joints, in_chans=3, embed_dim_ratio=512, depth=8,
        num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,drop_path_rate=0)
    elif args.backbone=='stgcn':
        from lib.model.model_stgcn import Model as STGCN 
        model_backbone = STGCN()
    else:
        raise Exception("Undefined backbone type.")
    return model_backbone

def masked_mse_loss(pred, target, joint_mask):
    """
    Compute the Mean Squared Error (MSE) with joint masking.

    Args:
        pred: Tensor of shape (batch, frames, joints, 3)
        target: Tensor of shape (batch, frames, joints, 3)
        joint_mask: Tensor of shape (joints,), with 1 for valid joints and 0 for ignored ones

    Returns:
        Scalar MSE loss over masked joints
    """
    assert pred.shape == target.shape, "Predicted and target shapes must match"
    
    # Reshape the mask to apply over batch, frames, and angle dimensions
    joint_mask = joint_mask.view(1, 1, -1, 1).to(pred.device)  # shape (1, 1, joints, 1)

    # Apply mask
    masked_pred = pred * joint_mask
    masked_target = target * joint_mask

    # Compute MSE manually
    mse = ((masked_pred - masked_target) ** 2).sum() / (joint_mask.sum() * pred.shape[0] * pred.shape[1] * pred.shape[-1])
    return mse

def mae_loss(pred_angles, true_angles, joint_mask):
    """
    Compute the Mean Angular Error (MAE) between predicted and true angles, applying a joint mask.

    Parameters:
    - pred_angles: torch.Tensor of shape (batch_size, num_frames, num_joints, 3), in degrees
    - true_angles: torch.Tensor of shape (batch_size, num_frames, num_joints, 3), in degrees
    - joint_mask: torch.Tensor of shape (num_joints,), mask for joints to include in the calculation
    
    Returns:
    - mae: Mean Angular Error as a tensor (retain gradients for backpropagation)
    """
    assert pred_angles.shape == true_angles.shape, "Shape mismatch between predicted and true angles."

    # Check if input has the shape (batch_size, num_joints, 3) or (batch_size, num_frames, num_joints, 3)
    if len(pred_angles.shape) == 3:  # Shape: (batch_size, num_joints, 3)
        num_frames = 1
        #pred_angles = pred_angles.unsqueeze(1)  # Add a frames dimension
        #true_angles = true_angles.unsqueeze(1)  # Add a frames dimension
    else:
        num_frames = pred_angles.shape[1]  # Extract the number of frames

    # Compute absolute angular difference
    angle_diff = torch.abs(pred_angles - true_angles)

    # Normalize angular differences to be within [0, 180] degrees
    angle_diff = torch.where(angle_diff > 180, 360 - angle_diff, angle_diff)

    # Reshape joint_mask for broadcasting across batch, frames, and angular components
    joint_mask = joint_mask.view(1, 1, -1, 1).to(angle_diff.device)  # Shape: (1, 1, num_joints, 1)

    # Apply joint mask
    masked_angle_diff = angle_diff * joint_mask

    # Total number of active elements (masked joints only)
    total_active_elements = joint_mask.sum() * pred_angles.shape[0] * num_frames

    # Compute MAE: Sum over all dimensions and normalize by active elements
    mae = masked_angle_diff.sum() / (total_active_elements * pred_angles.shape[-1])

    return mae

def mae(pred_angles, true_angles, joint_mask):
    """
    Compute the Mean Angular Error (MAE) between predicted and true angles, applying a joint mask.

    Parameters:
    - pred_angles: torch.Tensor of shape (batch_size, num_frames, num_joints, 3), in degrees
    - true_angles: torch.Tensor of shape (batch_size, num_frames, num_joints, 3), in degrees
    - joint_mask: torch.Tensor of shape (num_joints,), mask for joints to include in the calculation
    
    Returns:
    - mae: Mean Angular Error in degrees
    """
    assert pred_angles.shape == true_angles.shape, "Shape mismatch between predicted and true angles."

    # Check if input has the shape (batch_size, num_joints, 3) or (batch_size, num_frames, num_joints, 3)
    if len(pred_angles.shape) == 3:  # Shape: (batch_size, num_joints, 3)
        num_frames = 1
        #pred_angles = pred_angles.unsqueeze(1)  # Add a frames dimension
        #true_angles = true_angles.unsqueeze(1)  # Add a frames dimension
    else:
        num_frames = pred_angles.shape[1]  # Extract the number of frames

    # Compute absolute angular difference
    angle_diff = torch.abs(pred_angles - true_angles)

    # Normalize angular differences to be within [0, 180] degrees
    angle_diff = torch.where(angle_diff > 180, 360 - angle_diff, angle_diff)

    # Reshape joint_mask for broadcasting across batch, frames, and angular components
    joint_mask = joint_mask.view(1, 1, -1, 1).to(angle_diff.device)  # Shape: (1, 1, num_joints, 1)

    # Apply joint mask
    masked_angle_diff = angle_diff * joint_mask

    # Total number of active elements (masked joints only)
    total_active_elements = joint_mask.sum() * pred_angles.shape[0] * num_frames

    # Compute MAE: Sum over all dimensions and normalize by active elements
    mae = masked_angle_diff.sum() / (total_active_elements * pred_angles.shape[-1])

    return mae.item()
