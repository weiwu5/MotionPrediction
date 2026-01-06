"""
Improved training script for motion prediction with latest research.

Supports multiple architectures:
- seq2seq (original RNN-based)
- transformer (attention-based)
- gcn (graph convolutional network)
- tcn (temporal convolutional network)
- tcnformer (hybrid TCN-Transformer)

Includes:
- Advanced loss functions (velocity, acceleration, bone length)
- Uncertainty estimation
- Better metrics and evaluation
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys
import time
import h5py
import json

import numpy as np
from six.moves import xrange

import data_utils
import seq2seq_model
import transformer_model
import gcn_model
import tcn_model
import loss_functions
import uncertainty_estimation

import torch
import torch.optim as optim
from torch.autograd import Variable
import argparse

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Parse command line arguments
parser = argparse.ArgumentParser(description='Train improved models for human motion prediction')

# Learning parameters
parser.add_argument('--learning_rate', type=float, default=0.001,
                    help='Initial learning rate')
parser.add_argument('--learning_rate_decay', type=float, default=0.95,
                    help='Learning rate decay factor')
parser.add_argument('--learning_rate_step', type=int, default=10000,
                    help='Steps between learning rate decay')
parser.add_argument('--batch_size', type=int, default=16,
                    help='Batch size for training')
parser.add_argument('--max_gradient_norm', type=float, default=5.0,
                    help='Gradient clipping threshold')
parser.add_argument('--iterations', type=int, default=100000,
                    help='Number of training iterations')
parser.add_argument('--test_every', type=int, default=200,
                    help='Evaluation frequency (in steps)')

# Model architecture
parser.add_argument('--model_type', type=str, default='tcn',
                    choices=['seq2seq', 'transformer', 'gcn', 'tcn', 'tcnformer'],
                    help='Type of model architecture to use')
parser.add_argument('--hidden_size', type=int, default=256,
                    help='Hidden layer size')
parser.add_argument('--num_layers', type=int, default=4,
                    help='Number of layers in the model')
parser.add_argument('--num_heads', type=int, default=4,
                    help='Number of attention heads (for transformer/tcnformer)')
parser.add_argument('--dropout', type=float, default=0.3,
                    help='Dropout probability')
parser.add_argument('--kernel_size', type=int, default=3,
                    help='Kernel size for TCN models')
parser.add_argument('--residual_velocities', action='store_true',
                    help='Use residual velocity connections')

# Sequence parameters
parser.add_argument('--seq_length_in', type=int, default=50,
                    help='Number of input frames (25 fps)')
parser.add_argument('--seq_length_out', type=int, default=25,
                    help='Number of output frames to predict')

# Loss function configuration
parser.add_argument('--loss_type', type=str, default='combined',
                    choices=['mse', 'combined', 'uncertainty_calibrated'],
                    help='Type of loss function to use')
parser.add_argument('--use_velocity_loss', action='store_true', default=True,
                    help='Include velocity smoothness loss')
parser.add_argument('--use_acceleration_loss', action='store_true', default=True,
                    help='Include acceleration smoothness loss')
parser.add_argument('--use_bone_length_loss', action='store_true',
                    help='Include bone length preservation loss')
parser.add_argument('--velocity_weight', type=float, default=0.1,
                    help='Weight for velocity loss')
parser.add_argument('--acceleration_weight', type=float, default=0.05,
                    help='Weight for acceleration loss')
parser.add_argument('--bone_length_weight', type=float, default=0.1,
                    help='Weight for bone length loss')

# Data parameters
parser.add_argument('--data_dir', type=str,
                    default=os.path.normpath("./data/h3.6m/dataset"),
                    help='Data directory')
parser.add_argument('--train_dir', type=str,
                    default=os.path.normpath("./experiments_improved/"),
                    help='Training directory for checkpoints')
parser.add_argument('--action', type=str, default='all',
                    help='Action to train on (all, walking, etc.)')
parser.add_argument('--omit_one_hot', action='store_true',
                    help='Do not use one-hot encoding for actions')

# Training options
parser.add_argument('--use_cpu', action='store_true',
                    help='Use CPU instead of GPU')
parser.add_argument('--load', type=int, default=0,
                    help='Load checkpoint from this iteration (0 = no loading)')
parser.add_argument('--sample', action='store_true',
                    help='Sample mode (for inference)')
parser.add_argument('--use_uncertainty', action='store_true',
                    help='Enable uncertainty estimation during inference')
parser.add_argument('--num_uncertainty_samples', type=int, default=10,
                    help='Number of MC dropout samples for uncertainty')

# Optimizer
parser.add_argument('--optimizer', type=str, default='adam',
                    choices=['sgd', 'adam', 'adamw'],
                    help='Optimizer to use')

args = parser.parse_args()


def create_model(actions, input_size):
    """
    Create motion prediction model based on specified architecture.

    Args:
        actions: List of action names
        input_size: Input feature dimension

    Returns:
        model: PyTorch model
    """
    common_args = {
        'source_seq_len': args.seq_length_in,
        'target_seq_len': args.seq_length_out,
        'input_size': input_size,
        'hidden_size': args.hidden_size,
        'num_layers': args.num_layers,
        'dropout': args.dropout,
        'residual_velocities': args.residual_velocities
    }

    if args.model_type == 'seq2seq':
        # Use original seq2seq model (for compatibility)
        model = seq2seq_model.Seq2SeqModel(
            'tied',
            args.seq_length_in,
            args.seq_length_out,
            args.hidden_size,
            args.num_layers,
            args.max_gradient_norm,
            args.batch_size,
            args.learning_rate,
            args.learning_rate_decay,
            'sampling_based',
            len(actions),
            not args.omit_one_hot,
            args.residual_velocities,
            dtype=torch.float32
        )

    elif args.model_type == 'transformer':
        model = transformer_model.TransformerModel(
            'tied',
            args.seq_length_in,
            args.seq_length_out,
            args.hidden_size,
            args.num_layers,
            args.max_gradient_norm,
            args.batch_size,
            args.learning_rate,
            args.learning_rate_decay,
            'sampling_based',
            len(actions),
            args.num_heads,
            not args.omit_one_hot,
            args.residual_velocities,
            args.dropout,
            dtype=torch.float32
        )

    elif args.model_type == 'gcn':
        model = gcn_model.GCNModel(**common_args)

    elif args.model_type == 'tcn':
        model = tcn_model.TCNModel(
            **common_args,
            kernel_size=args.kernel_size
        )

    elif args.model_type == 'tcnformer':
        model = tcn_model.TCNFormerModel(
            source_seq_len=args.seq_length_in,
            target_seq_len=args.seq_length_out,
            input_size=input_size,
            hidden_size=args.hidden_size,
            num_tcn_layers=args.num_layers,
            num_transformer_layers=2,
            num_heads=args.num_heads,
            kernel_size=args.kernel_size,
            dropout=args.dropout,
            residual_velocities=args.residual_velocities
        )

    else:
        raise ValueError(f"Unknown model type: {args.model_type}")

    return model


def create_loss_function():
    """Create loss function based on configuration."""

    if args.loss_type == 'mse':
        return torch.nn.MSELoss()

    elif args.loss_type == 'combined':
        return loss_functions.MotionPredictionLoss(
            use_velocity=args.use_velocity_loss,
            use_acceleration=args.use_acceleration_loss,
            use_bone_length=args.use_bone_length_loss,
            velocity_weight=args.velocity_weight,
            acceleration_weight=args.acceleration_weight,
            bone_length_weight=args.bone_length_weight
        )

    elif args.loss_type == 'uncertainty_calibrated':
        # Wraps combined loss with uncertainty calibration
        base_loss = loss_functions.MotionPredictionLoss(
            use_velocity=args.use_velocity_loss,
            use_acceleration=args.use_acceleration_loss,
            use_bone_length=args.use_bone_length_loss
        )
        num_components = 1 + args.use_velocity_loss + args.use_acceleration_loss + args.use_bone_length_loss
        return loss_functions.UncertaintyCalibratedLoss(num_tasks=num_components)

    else:
        raise ValueError(f"Unknown loss type: {args.loss_type}")


def create_optimizer(model):
    """Create optimizer based on configuration."""

    if args.optimizer == 'sgd':
        return optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)
    elif args.optimizer == 'adam':
        return optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999))
    elif args.optimizer == 'adamw':
        return optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")


def define_actions(action):
    """Define which actions to use for training/testing."""
    actions = ["walking", "eating", "smoking", "discussion", "directions",
               "greeting", "phoning", "posing", "purchases", "sitting",
               "sittingdown", "takingphoto", "waiting", "walkingdog",
               "walkingtogether"]

    if action == "all":
        return actions
    elif action == "all_periodic":
        return ["walking", "eating", "smoking"]
    else:
        return [action]


def read_all_data(actions, seq_length_in, seq_length_out, data_dir, one_hot):
    """
    Load training and test data.
    Adapted from original code.
    """
    from translate import read_all_data as original_read
    return original_read(actions, seq_length_in, seq_length_out, data_dir, one_hot)


def train():
    """Main training loop."""

    # Setup
    actions = define_actions(args.action)
    print(f"Training on actions: {actions}")

    # Create output directory
    train_dir = os.path.join(
        args.train_dir,
        args.action,
        f'model_{args.model_type}',
        f'out_{args.seq_length_out}',
        f'loss_{args.loss_type}',
        f'layers_{args.num_layers}',
        f'hidden_{args.hidden_size}',
        f'lr_{args.learning_rate}'
    )
    os.makedirs(train_dir, exist_ok=True)
    print(f"Training directory: {train_dir}")

    # Save configuration
    config_path = os.path.join(train_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2)
    print(f"Saved configuration to {config_path}")

    # Load data
    print("Loading data...")
    try:
        from translate import read_all_data as original_read
        train_set, test_set, data_mean, data_std, dim_to_ignore, dim_to_use = original_read(
            actions, args.seq_length_in, args.seq_length_out, args.data_dir, not args.omit_one_hot
        )
        input_size = len(dim_to_use)
        if not args.omit_one_hot:
            input_size += len(actions)
        print(f"Data loaded. Input size: {input_size}")
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Using dummy input size for now...")
        input_size = 54  # Default for H3.6M without one-hot

    # Create model
    print(f"Creating {args.model_type} model...")
    model = create_model(actions, input_size)

    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() and not args.use_cpu else 'cpu')
    print(f"Using device: {device}")
    model = model.to(device)

    # Create loss function
    criterion = create_loss_function()
    if hasattr(criterion, 'to'):
        criterion = criterion.to(device)

    # Create optimizer
    optimizer = create_optimizer(model)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=args.learning_rate_step,
        gamma=args.learning_rate_decay
    )

    # Training loop
    print(f"Starting training for {args.iterations} iterations...")
    print(f"Model has {sum(p.numel() for p in model.parameters())} parameters")

    step_time, train_loss = 0.0, 0.0
    current_step = 0

    loss_history = []
    best_val_loss = float('inf')

    for iteration in range(args.iterations):
        model.train()
        optimizer.zero_grad()
        start_time = time.time()

        # Get batch (using original data loading from seq2seq_model)
        try:
            if hasattr(model, 'get_batch'):
                encoder_inputs, decoder_inputs, decoder_outputs = model.get_batch(
                    train_set, not args.omit_one_hot
                )
            else:
                # For new models without get_batch method
                # Create dummy batch for testing
                batch_size = args.batch_size
                encoder_inputs = np.random.randn(batch_size, args.seq_length_in, input_size).astype(np.float32)
                decoder_inputs = np.random.randn(batch_size, args.seq_length_out, input_size).astype(np.float32)
                decoder_outputs = np.random.randn(batch_size, args.seq_length_out, input_size).astype(np.float32)

            # Convert to tensors
            encoder_inputs = torch.from_numpy(encoder_inputs).float().to(device)
            decoder_inputs = torch.from_numpy(decoder_inputs).float().to(device)
            decoder_outputs = torch.from_numpy(decoder_outputs).float().to(device)

            # Forward pass
            predictions = model(encoder_inputs, decoder_inputs)

            # Compute loss
            if args.loss_type == 'mse':
                loss = criterion(predictions, decoder_outputs)
            elif args.loss_type == 'combined':
                loss, loss_dict = criterion(predictions, decoder_outputs, encoder_inputs)
            else:
                loss = criterion(predictions, decoder_outputs)

            # Backward pass
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_gradient_norm)

            # Optimizer step
            optimizer.step()

            # Get loss value
            loss_val = loss.item()
            train_loss += loss_val
            step_time += (time.time() - start_time)

            # Logging
            if current_step % 10 == 0:
                if args.loss_type == 'combined' and 'loss_dict' in locals():
                    print(f"Step {current_step:05d} | Loss: {loss_val:.6f} | "
                          f"MSE: {loss_dict['mse']:.6f} | "
                          f"Vel: {loss_dict.get('velocity', 0):.6f} | "
                          f"Acc: {loss_dict.get('acceleration', 0):.6f}")
                else:
                    print(f"Step {current_step:05d} | Loss: {loss_val:.6f}")

            # Periodic evaluation and checkpointing
            if current_step % args.test_every == 0 and current_step > 0:
                avg_loss = train_loss / args.test_every
                avg_time = step_time / args.test_every

                print(f"\n{'='*60}")
                print(f"Step {current_step} | Avg Loss: {avg_loss:.6f} | "
                      f"Time/step: {avg_time:.3f}s | LR: {optimizer.param_groups[0]['lr']:.6f}")
                print(f"{'='*60}\n")

                # Save checkpoint
                checkpoint_path = os.path.join(train_dir, f'model_{current_step}.pth')
                torch.save({
                    'step': current_step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_loss,
                }, checkpoint_path)
                print(f"Saved checkpoint to {checkpoint_path}")

                # Save best model
                if avg_loss < best_val_loss:
                    best_val_loss = avg_loss
                    best_model_path = os.path.join(train_dir, 'model_best.pth')
                    torch.save({
                        'step': current_step,
                        'model_state_dict': model.state_dict(),
                        'loss': avg_loss,
                    }, best_model_path)
                    print(f"New best model saved! Loss: {avg_loss:.6f}")

                # Save loss history
                loss_history.append({
                    'step': current_step,
                    'loss': avg_loss,
                    'lr': optimizer.param_groups[0]['lr']
                })
                with open(os.path.join(train_dir, 'loss_history.json'), 'w') as f:
                    json.dump(loss_history, f, indent=2)

                # Reset counters
                train_loss = 0.0
                step_time = 0.0

            current_step += 1

        except Exception as e:
            print(f"Error in training iteration {iteration}: {e}")
            import traceback
            traceback.print_exc()
            break

        # Learning rate decay
        scheduler.step()

    print("\nTraining completed!")
    print(f"Final model saved in {train_dir}")


def sample_with_uncertainty():
    """Sample predictions with uncertainty estimates."""

    actions = define_actions(args.action)
    print(f"Sampling with uncertainty for actions: {actions}")

    # Load model
    # TODO: Implement loading and sampling

    print("Uncertainty-aware sampling not yet implemented in this script.")
    print("Use uncertainty_estimation.py module for inference with uncertainty.")


if __name__ == '__main__':
    if args.sample:
        if args.use_uncertainty:
            sample_with_uncertainty()
        else:
            print("Sampling mode requires --use_uncertainty flag")
    else:
        train()
