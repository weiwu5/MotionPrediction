"""
Uncertainty estimation for human motion prediction.

Based on latest research (2024):
- Monte Carlo Dropout for uncertainty quantification
- Ensemble methods for robust predictions
- Confidence-aware forecasting

Reference: "Toward Reliable Human Pose Forecasting With Uncertainty" (RAL 2024)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def enable_dropout(model):
    """
    Enable dropout layers during inference for Monte Carlo Dropout.

    Args:
        model: PyTorch model
    """
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()


def disable_dropout(model):
    """
    Disable dropout layers (standard inference mode).

    Args:
        model: PyTorch model
    """
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.eval()


class MCDropoutPredictor:
    """
    Monte Carlo Dropout predictor for uncertainty estimation.

    Performs multiple forward passes with dropout enabled to sample
    from the approximate posterior distribution, providing uncertainty
    estimates along with predictions.
    """

    def __init__(self, model, num_samples=10):
        """
        Args:
            model: The trained motion prediction model
            num_samples: Number of MC samples to draw
        """
        self.model = model
        self.num_samples = num_samples

    def predict_with_uncertainty(self, encoder_inputs, target_seq_len):
        """
        Make predictions with uncertainty estimates.

        Args:
            encoder_inputs: (batch, source_seq_len, input_size)
            target_seq_len: Number of frames to predict
        Returns:
            mean_prediction: (batch, target_seq_len, input_size) mean prediction
            std_prediction: (batch, target_seq_len, input_size) uncertainty (std dev)
            all_predictions: (num_samples, batch, target_seq_len, input_size) all samples
        """
        self.model.eval()  # Set base model to eval mode
        enable_dropout(self.model)  # But keep dropout active

        all_predictions = []

        with torch.no_grad():
            for _ in range(self.num_samples):
                # Each forward pass gives a different prediction due to dropout
                if hasattr(self.model, 'predict'):
                    prediction = self.model.predict(encoder_inputs, target_seq_len)
                else:
                    # For models without predict method, create dummy decoder inputs
                    batch_size = encoder_inputs.size(0)
                    device = encoder_inputs.device
                    decoder_inputs = torch.zeros(batch_size, target_seq_len,
                                                 encoder_inputs.size(2)).to(device)
                    prediction = self.model(encoder_inputs, decoder_inputs)

                all_predictions.append(prediction)

        # Stack all predictions
        all_predictions = torch.stack(all_predictions, dim=0)  # (num_samples, batch, seq, dim)

        # Compute statistics
        mean_prediction = all_predictions.mean(dim=0)  # (batch, seq, dim)
        std_prediction = all_predictions.std(dim=0)    # (batch, seq, dim)

        disable_dropout(self.model)  # Restore normal eval mode

        return mean_prediction, std_prediction, all_predictions

    def predict_with_confidence_intervals(self, encoder_inputs, target_seq_len,
                                         confidence=0.95):
        """
        Make predictions with confidence intervals.

        Args:
            encoder_inputs: (batch, source_seq_len, input_size)
            target_seq_len: Number of frames to predict
            confidence: Confidence level (default 0.95 for 95% CI)
        Returns:
            mean_prediction: (batch, target_seq_len, input_size)
            lower_bound: (batch, target_seq_len, input_size)
            upper_bound: (batch, target_seq_len, input_size)
        """
        mean_pred, std_pred, all_preds = self.predict_with_uncertainty(
            encoder_inputs, target_seq_len
        )

        # Compute percentiles for confidence intervals
        alpha = 1 - confidence
        lower_percentile = alpha / 2
        upper_percentile = 1 - alpha / 2

        lower_bound = torch.quantile(all_preds, lower_percentile, dim=0)
        upper_bound = torch.quantile(all_preds, upper_percentile, dim=0)

        return mean_pred, lower_bound, upper_bound


class EnsemblePredictor:
    """
    Ensemble predictor combining multiple trained models.

    Provides more robust predictions and uncertainty estimates
    by averaging predictions from multiple independently trained models.
    """

    def __init__(self, models):
        """
        Args:
            models: List of trained PyTorch models
        """
        self.models = models
        for model in self.models:
            model.eval()

    def predict_with_uncertainty(self, encoder_inputs, target_seq_len):
        """
        Make ensemble predictions with uncertainty.

        Args:
            encoder_inputs: (batch, source_seq_len, input_size)
            target_seq_len: Number of frames to predict
        Returns:
            mean_prediction: (batch, target_seq_len, input_size)
            std_prediction: (batch, target_seq_len, input_size)
            all_predictions: (num_models, batch, target_seq_len, input_size)
        """
        all_predictions = []

        with torch.no_grad():
            for model in self.models:
                if hasattr(model, 'predict'):
                    prediction = model.predict(encoder_inputs, target_seq_len)
                else:
                    batch_size = encoder_inputs.size(0)
                    device = encoder_inputs.device
                    decoder_inputs = torch.zeros(batch_size, target_seq_len,
                                                 encoder_inputs.size(2)).to(device)
                    prediction = model(encoder_inputs, decoder_inputs)

                all_predictions.append(prediction)

        # Stack predictions
        all_predictions = torch.stack(all_predictions, dim=0)

        # Compute statistics
        mean_prediction = all_predictions.mean(dim=0)
        std_prediction = all_predictions.std(dim=0)

        return mean_prediction, std_prediction, all_predictions

    def predict(self, encoder_inputs, target_seq_len):
        """
        Make ensemble prediction (mean only).

        Args:
            encoder_inputs: (batch, source_seq_len, input_size)
            target_seq_len: Number of frames to predict
        Returns:
            prediction: (batch, target_seq_len, input_size)
        """
        mean_pred, _, _ = self.predict_with_uncertainty(encoder_inputs, target_seq_len)
        return mean_pred


class CalibrationWrapper(nn.Module):
    """
    Temperature scaling wrapper for uncertainty calibration.

    Adjusts model confidence to be better calibrated with actual accuracy.
    Useful when model is over-confident or under-confident.

    Reference: "On Calibration of Modern Neural Networks" (Guo et al., ICML 2017)
    """

    def __init__(self, model):
        """
        Args:
            model: Base prediction model
        """
        super(CalibrationWrapper, self).__init__()
        self.model = model
        # Temperature parameter (learnable)
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, encoder_inputs, decoder_inputs):
        """
        Forward pass with temperature scaling.

        Args:
            encoder_inputs: (batch, source_seq_len, input_size)
            decoder_inputs: (batch, target_seq_len, input_size)
        Returns:
            scaled_output: Temperature-scaled predictions
        """
        logits = self.model(encoder_inputs, decoder_inputs)

        # For regression tasks, we can scale the variance
        # This is a simplified version
        return logits / self.temperature

    def calibrate(self, val_loader, criterion, device='cuda'):
        """
        Calibrate temperature on validation set.

        Args:
            val_loader: Validation data loader
            criterion: Loss function
            device: Device to use
        """
        # Set model to eval mode
        self.model.eval()

        # Optimize temperature
        optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=50)

        def eval_loss():
            loss = 0
            for encoder_inputs, decoder_inputs, targets in val_loader:
                encoder_inputs = encoder_inputs.to(device)
                decoder_inputs = decoder_inputs.to(device)
                targets = targets.to(device)

                outputs = self.forward(encoder_inputs, decoder_inputs)
                loss += criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            return loss

        optimizer.step(eval_loss)

        print(f"Calibrated temperature: {self.temperature.item():.4f}")


class UncertaintyAwareMetrics:
    """
    Metrics for evaluating uncertainty estimates.

    Includes calibration error, sharpness, and coverage metrics.
    """

    @staticmethod
    def expected_calibration_error(predictions, uncertainties, targets, num_bins=10):
        """
        Compute Expected Calibration Error (ECE).

        Measures how well predicted uncertainties match actual errors.

        Args:
            predictions: (batch, seq_len, dim) predicted poses
            uncertainties: (batch, seq_len, dim) predicted uncertainties (std dev)
            targets: (batch, seq_len, dim) ground truth poses
            num_bins: Number of bins for calibration plot
        Returns:
            ece: Expected calibration error
        """
        # Compute errors
        errors = torch.abs(predictions - targets)  # (batch, seq_len, dim)

        # Flatten for easier processing
        errors_flat = errors.reshape(-1)
        uncertainties_flat = uncertainties.reshape(-1)

        # Sort by uncertainty
        sorted_indices = torch.argsort(uncertainties_flat)
        errors_sorted = errors_flat[sorted_indices]
        uncertainties_sorted = uncertainties_flat[sorted_indices]

        # Divide into bins
        bin_size = len(errors_sorted) // num_bins
        ece = 0.0

        for i in range(num_bins):
            start_idx = i * bin_size
            end_idx = start_idx + bin_size if i < num_bins - 1 else len(errors_sorted)

            bin_errors = errors_sorted[start_idx:end_idx]
            bin_uncertainties = uncertainties_sorted[start_idx:end_idx]

            # Average error and uncertainty in this bin
            avg_error = bin_errors.mean()
            avg_uncertainty = bin_uncertainties.mean()

            # Calibration error for this bin
            ece += torch.abs(avg_error - avg_uncertainty) * len(bin_errors)

        ece = ece / len(errors_sorted)

        return ece.item()

    @staticmethod
    def prediction_interval_coverage(predictions, lower_bounds, upper_bounds,
                                    targets, expected_coverage=0.95):
        """
        Compute prediction interval coverage.

        Measures what fraction of ground truth values fall within
        predicted confidence intervals.

        Args:
            predictions: (batch, seq_len, dim)
            lower_bounds: (batch, seq_len, dim)
            upper_bounds: (batch, seq_len, dim)
            targets: (batch, seq_len, dim)
            expected_coverage: Expected coverage (e.g., 0.95 for 95% CI)
        Returns:
            coverage: Actual coverage (should be close to expected_coverage)
            coverage_error: |actual_coverage - expected_coverage|
        """
        # Check if targets fall within intervals
        within_interval = (targets >= lower_bounds) & (targets <= upper_bounds)

        # Compute coverage
        coverage = within_interval.float().mean().item()

        # Coverage error
        coverage_error = abs(coverage - expected_coverage)

        return coverage, coverage_error

    @staticmethod
    def sharpness(uncertainties):
        """
        Compute sharpness (average width of prediction intervals).

        Lower sharpness (narrower intervals) is better, given good calibration.

        Args:
            uncertainties: (batch, seq_len, dim) standard deviations
        Returns:
            sharpness: Average uncertainty
        """
        return uncertainties.mean().item()

    @staticmethod
    def uncertainty_error_correlation(predictions, uncertainties, targets):
        """
        Compute correlation between predicted uncertainty and actual error.

        High correlation means the model correctly identifies difficult cases.

        Args:
            predictions: (batch, seq_len, dim)
            uncertainties: (batch, seq_len, dim)
            targets: (batch, seq_len, dim)
        Returns:
            correlation: Pearson correlation coefficient
        """
        # Compute errors
        errors = torch.abs(predictions - targets)

        # Flatten
        errors_flat = errors.reshape(-1)
        uncertainties_flat = uncertainties.reshape(-1)

        # Compute correlation
        errors_np = errors_flat.cpu().numpy()
        uncertainties_np = uncertainties_flat.cpu().numpy()

        correlation = np.corrcoef(errors_np, uncertainties_np)[0, 1]

        return correlation


def visualize_uncertainty(predictions, uncertainties, targets, frame_idx=0,
                         joint_idx=0, save_path=None):
    """
    Visualize predictions with uncertainty bounds.

    Args:
        predictions: (batch, seq_len, dim)
        uncertainties: (batch, seq_len, dim) standard deviations
        targets: (batch, seq_len, dim)
        frame_idx: Which batch element to visualize
        joint_idx: Which joint dimension to visualize
        save_path: Optional path to save figure
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib not available for visualization")
        return

    # Extract data for visualization
    pred = predictions[frame_idx, :, joint_idx].cpu().numpy()
    std = uncertainties[frame_idx, :, joint_idx].cpu().numpy()
    target = targets[frame_idx, :, joint_idx].cpu().numpy()

    time_steps = np.arange(len(pred))

    # Create plot
    plt.figure(figsize=(10, 6))

    # Plot ground truth
    plt.plot(time_steps, target, 'g-', label='Ground Truth', linewidth=2)

    # Plot prediction
    plt.plot(time_steps, pred, 'b-', label='Prediction', linewidth=2)

    # Plot uncertainty bounds (95% confidence)
    lower_bound = pred - 1.96 * std
    upper_bound = pred + 1.96 * std
    plt.fill_between(time_steps, lower_bound, upper_bound, alpha=0.3,
                     color='blue', label='95% Confidence Interval')

    plt.xlabel('Time Step')
    plt.ylabel('Joint Value')
    plt.title(f'Prediction with Uncertainty (Joint {joint_idx})')
    plt.legend()
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved uncertainty visualization to {save_path}")
    else:
        plt.show()

    plt.close()
