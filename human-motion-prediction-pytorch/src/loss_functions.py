"""
Improved loss functions for human motion prediction.

Based on latest research (2024-2025):
- Physical constraints (bone lengths preservation)
- Velocity smoothness
- Acceleration smoothness
- Poses realistic and physically plausible

These losses help produce more realistic and physically valid motion predictions.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MotionPredictionLoss(nn.Module):
    """
    Combined loss function for motion prediction with multiple components:

    1. MSE Loss: Basic reconstruction loss
    2. Velocity Loss: Encourages smooth velocity transitions
    3. Acceleration Loss: Penalizes sudden jerky movements
    4. Bone Length Loss: Preserves skeleton structure
    5. Foot Contact Loss: Prevents foot sliding

    Each component can be weighted differently.
    """

    def __init__(self, use_velocity=True, use_acceleration=True,
                 use_bone_length=False, use_foot_contact=False,
                 mse_weight=1.0, velocity_weight=0.1, acceleration_weight=0.05,
                 bone_length_weight=0.1, foot_contact_weight=0.1):
        """
        Args:
            use_velocity: Include velocity smoothness loss
            use_acceleration: Include acceleration smoothness loss
            use_bone_length: Include bone length preservation loss
            use_foot_contact: Include foot contact consistency loss
            *_weight: Weight for each loss component
        """
        super(MotionPredictionLoss, self).__init__()

        self.use_velocity = use_velocity
        self.use_acceleration = use_acceleration
        self.use_bone_length = use_bone_length
        self.use_foot_contact = use_foot_contact

        self.mse_weight = mse_weight
        self.velocity_weight = velocity_weight
        self.acceleration_weight = acceleration_weight
        self.bone_length_weight = bone_length_weight
        self.foot_contact_weight = foot_contact_weight

    def forward(self, predictions, targets, encoder_inputs=None):
        """
        Compute combined loss.

        Args:
            predictions: (batch, seq_len, input_size) predicted poses
            targets: (batch, seq_len, input_size) ground truth poses
            encoder_inputs: (batch, source_seq_len, input_size) historical poses (optional)
        Returns:
            loss: scalar tensor
            loss_dict: dictionary with individual loss components
        """
        loss_dict = {}

        # 1. MSE Loss
        mse_loss = F.mse_loss(predictions, targets)
        loss_dict['mse'] = mse_loss.item()
        total_loss = self.mse_weight * mse_loss

        # 2. Velocity Loss
        if self.use_velocity:
            velocity_loss = self.compute_velocity_loss(predictions, targets, encoder_inputs)
            loss_dict['velocity'] = velocity_loss.item()
            total_loss = total_loss + self.velocity_weight * velocity_loss

        # 3. Acceleration Loss
        if self.use_acceleration:
            acceleration_loss = self.compute_acceleration_loss(predictions, targets, encoder_inputs)
            loss_dict['acceleration'] = acceleration_loss.item()
            total_loss = total_loss + self.acceleration_weight * acceleration_loss

        # 4. Bone Length Loss (requires 3D joint positions)
        if self.use_bone_length:
            bone_loss = self.compute_bone_length_loss(predictions, targets)
            loss_dict['bone_length'] = bone_loss.item()
            total_loss = total_loss + self.bone_length_weight * bone_loss

        # 5. Foot Contact Loss
        if self.use_foot_contact:
            foot_loss = self.compute_foot_contact_loss(predictions, targets)
            loss_dict['foot_contact'] = foot_loss.item()
            total_loss = total_loss + self.foot_contact_weight * foot_loss

        loss_dict['total'] = total_loss.item()

        return total_loss, loss_dict

    def compute_velocity_loss(self, predictions, targets, encoder_inputs=None):
        """
        Compute velocity smoothness loss.

        Encourages smooth transitions in motion by penalizing differences
        between predicted and ground truth velocities.

        Args:
            predictions: (batch, seq_len, input_size)
            targets: (batch, seq_len, input_size)
            encoder_inputs: (batch, source_seq_len, input_size) optional
        Returns:
            velocity_loss: scalar tensor
        """
        # Compute velocities (first-order differences)
        pred_velocity = predictions[:, 1:, :] - predictions[:, :-1, :]
        target_velocity = targets[:, 1:, :] - targets[:, :-1, :]

        # MSE on velocities
        velocity_loss = F.mse_loss(pred_velocity, target_velocity)

        # Also check continuity with last encoder input if available
        if encoder_inputs is not None:
            last_encoder_pose = encoder_inputs[:, -1:, :]  # (batch, 1, input_size)
            first_pred_pose = predictions[:, 0:1, :]  # (batch, 1, input_size)

            # Velocity from last encoder to first prediction
            transition_velocity_pred = first_pred_pose - last_encoder_pose

            # Compare with ground truth transition
            first_target_pose = targets[:, 0:1, :]
            transition_velocity_target = first_target_pose - last_encoder_pose

            transition_loss = F.mse_loss(transition_velocity_pred, transition_velocity_target)
            velocity_loss = velocity_loss + transition_loss

        return velocity_loss

    def compute_acceleration_loss(self, predictions, targets, encoder_inputs=None):
        """
        Compute acceleration smoothness loss.

        Penalizes jerky motion by comparing second-order differences
        (acceleration) between predictions and ground truth.

        Args:
            predictions: (batch, seq_len, input_size)
            targets: (batch, seq_len, input_size)
            encoder_inputs: (batch, source_seq_len, input_size) optional
        Returns:
            acceleration_loss: scalar tensor
        """
        # Compute accelerations (second-order differences)
        if predictions.size(1) < 3:
            # Need at least 3 frames for acceleration
            return torch.tensor(0.0, device=predictions.device)

        # Velocity
        pred_velocity = predictions[:, 1:, :] - predictions[:, :-1, :]
        target_velocity = targets[:, 1:, :] - targets[:, :-1, :]

        # Acceleration
        pred_acceleration = pred_velocity[:, 1:, :] - pred_velocity[:, :-1, :]
        target_acceleration = target_velocity[:, 1:, :] - target_velocity[:, :-1, :]

        # MSE on accelerations
        acceleration_loss = F.mse_loss(pred_acceleration, target_acceleration)

        return acceleration_loss

    def compute_bone_length_loss(self, predictions, targets):
        """
        Compute bone length preservation loss.

        Ensures that predicted poses maintain consistent bone lengths
        throughout the sequence, preserving the skeleton structure.

        Note: This assumes the input is in a format where we can extract
        3D joint positions. For exponential map representation, this is
        an approximation based on angle differences.

        Args:
            predictions: (batch, seq_len, input_size)
            targets: (batch, seq_len, input_size)
        Returns:
            bone_length_loss: scalar tensor
        """
        # For exponential map representation, we approximate bone length
        # consistency by ensuring similar joint angle patterns

        # This is a simplified version - ideally would convert to 3D positions first
        batch_size, seq_len, input_size = predictions.shape

        # Reshape to separate joints (assuming 3 values per joint)
        num_joints = input_size // 3
        pred_joints = predictions.view(batch_size, seq_len, num_joints, 3)
        target_joints = targets.view(batch_size, seq_len, num_joints, 3)

        # Compute pairwise distances between joints
        # This is an approximation in exponential map space
        bone_loss = torch.tensor(0.0, device=predictions.device)

        # For each adjacent joint pair, check if relative distances are preserved
        for i in range(num_joints - 1):
            pred_diff = torch.norm(pred_joints[:, :, i, :] - pred_joints[:, :, i+1, :], dim=2)
            target_diff = torch.norm(target_joints[:, :, i, :] - target_joints[:, :, i+1, :], dim=2)

            # Penalize changes in bone lengths
            bone_loss = bone_loss + F.mse_loss(pred_diff, target_diff)

        bone_loss = bone_loss / (num_joints - 1)

        return bone_loss

    def compute_foot_contact_loss(self, predictions, targets):
        """
        Compute foot contact consistency loss.

        Helps prevent foot sliding artifacts by encouraging the model
        to maintain consistent foot positions when they should be in contact
        with the ground.

        This is a simplified version that checks velocity of foot joints.

        Args:
            predictions: (batch, seq_len, input_size)
            targets: (batch, seq_len, input_size)
        Returns:
            foot_contact_loss: scalar tensor
        """
        # For Human3.6M, foot joints are typically at specific indices
        # This is a simplified approximation

        # Assuming input_size = 54 (18 joints * 3)
        # Left foot ~= joint indices [5, 6] and right foot ~= [11, 12]
        # In flattened form: left foot = [15:21], right foot = [33:39]

        if predictions.size(2) < 54:
            return torch.tensor(0.0, device=predictions.device)

        # Extract foot joint features (approximate)
        left_foot_pred = predictions[:, :, 15:21]  # 2 joints * 3 dims
        left_foot_target = targets[:, :, 15:21]

        right_foot_pred = predictions[:, :, 33:39]
        right_foot_target = targets[:, :, 33:39]

        # Compute foot velocities
        left_foot_vel_pred = left_foot_pred[:, 1:, :] - left_foot_pred[:, :-1, :]
        left_foot_vel_target = left_foot_target[:, 1:, :] - left_foot_target[:, :-1, :]

        right_foot_vel_pred = right_foot_pred[:, 1:, :] - right_foot_pred[:, :-1, :]
        right_foot_vel_target = right_foot_target[:, 1:, :] - right_foot_target[:, :-1, :]

        # When ground truth foot velocity is small (foot on ground),
        # predicted foot should also have small velocity
        threshold = 0.01
        left_contact_mask = (torch.norm(left_foot_vel_target, dim=2) < threshold).float()
        right_contact_mask = (torch.norm(right_foot_vel_target, dim=2) < threshold).float()

        # Apply masked loss
        left_loss = (left_contact_mask.unsqueeze(2) *
                     (left_foot_vel_pred - left_foot_vel_target) ** 2).mean()
        right_loss = (right_contact_mask.unsqueeze(2) *
                      (right_foot_vel_pred - right_foot_vel_target) ** 2).mean()

        foot_contact_loss = (left_loss + right_loss) / 2

        return foot_contact_loss


class UncertaintyCalibratedLoss(nn.Module):
    """
    Loss function with learned uncertainty weighting.

    Automatically learns to balance multiple loss components
    by modeling task-dependent uncertainty (heteroscedastic uncertainty).

    Based on: "Multi-Task Learning Using Uncertainty to Weigh Losses
    for Scene Geometry and Semantics" (Kendall et al., CVPR 2018)
    """

    def __init__(self, num_tasks=3):
        """
        Args:
            num_tasks: Number of loss components to balance
        """
        super(UncertaintyCalibratedLoss, self).__init__()

        # Learnable log variance parameters (one per task)
        # Using log variance for numerical stability
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))

    def forward(self, losses):
        """
        Compute uncertainty-weighted loss.

        Args:
            losses: List or tensor of individual loss components [loss1, loss2, ...]
        Returns:
            total_loss: Weighted combination of losses
            weights: The learned weights (1 / (2 * sigma^2))
        """
        if isinstance(losses, list):
            losses = torch.stack(losses)

        # Uncertainty weighting: loss_i / (2 * sigma_i^2) + log(sigma_i)
        # Using log_var = log(sigma^2) for stability
        precision = torch.exp(-self.log_vars)  # 1 / sigma^2
        weighted_losses = precision * losses + self.log_vars

        total_loss = weighted_losses.sum()

        # Return weights for monitoring
        weights = 0.5 * precision

        return total_loss, weights


def compute_mpjpe(predictions, targets, align_root=True):
    """
    Compute Mean Per Joint Position Error (MPJPE).

    This is a standard evaluation metric for 3D pose estimation.

    Args:
        predictions: (batch, seq_len, num_joints * 3) predicted 3D positions
        targets: (batch, seq_len, num_joints * 3) ground truth 3D positions
        align_root: If True, align root joint (hip) before computing error
    Returns:
        mpjpe: Mean error in millimeters (or same unit as input)
    """
    batch_size, seq_len, dim = predictions.shape
    num_joints = dim // 3

    # Reshape to (batch, seq_len, num_joints, 3)
    pred_joints = predictions.view(batch_size, seq_len, num_joints, 3)
    target_joints = targets.view(batch_size, seq_len, num_joints, 3)

    if align_root:
        # Subtract root joint (joint 0) to align
        root_pred = pred_joints[:, :, 0:1, :]
        root_target = target_joints[:, :, 0:1, :]

        pred_joints = pred_joints - root_pred
        target_joints = target_joints - root_target

    # Compute per-joint errors
    errors = torch.norm(pred_joints - target_joints, dim=3)  # (batch, seq_len, num_joints)

    # Mean over all joints and frames
    mpjpe = errors.mean()

    return mpjpe


def compute_pck(predictions, targets, threshold=150.0):
    """
    Compute Percentage of Correct Keypoints (PCK).

    A joint is considered correct if it's within threshold distance
    from the ground truth position.

    Args:
        predictions: (batch, seq_len, num_joints * 3)
        targets: (batch, seq_len, num_joints * 3)
        threshold: Distance threshold in mm (default 150mm)
    Returns:
        pck: Percentage of correct keypoints (0-100)
    """
    batch_size, seq_len, dim = predictions.shape
    num_joints = dim // 3

    # Reshape to (batch, seq_len, num_joints, 3)
    pred_joints = predictions.view(batch_size, seq_len, num_joints, 3)
    target_joints = targets.view(batch_size, seq_len, num_joints, 3)

    # Compute per-joint errors
    errors = torch.norm(pred_joints - target_joints, dim=3)

    # Count correct predictions
    correct = (errors < threshold).float()

    # Percentage
    pck = correct.mean() * 100.0

    return pck


def compute_diversity_score(predictions_list):
    """
    Compute diversity score for stochastic predictions.

    Measures the diversity of multiple prediction samples
    (useful for generative models).

    Args:
        predictions_list: List of (batch, seq_len, input_size) tensors
                         Each element is a different sample
    Returns:
        diversity: Average pairwise distance between samples
    """
    num_samples = len(predictions_list)

    if num_samples < 2:
        return torch.tensor(0.0)

    # Stack all samples
    all_predictions = torch.stack(predictions_list, dim=0)  # (num_samples, batch, seq_len, input_size)

    # Compute pairwise distances
    total_distance = 0.0
    num_pairs = 0

    for i in range(num_samples):
        for j in range(i + 1, num_samples):
            distance = F.mse_loss(all_predictions[i], all_predictions[j])
            total_distance += distance
            num_pairs += 1

    diversity = total_distance / num_pairs if num_pairs > 0 else torch.tensor(0.0)

    return diversity
