"""
Temporal Convolutional Network (TCN) for human motion prediction.

Based on recent research (2024):
- Uses dilated causal convolutions for large receptive fields
- More efficient than RNNs while maintaining long-term dependencies
- Residual connections for better gradient flow
- Competitive performance with transformer models

Reference: WACV 2024, TCNFormer papers
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm


class TemporalBlock(nn.Module):
    """
    Temporal convolutional block with dilated convolutions.

    Key features:
    - Causal convolution (no future information leakage)
    - Dilation for exponentially growing receptive field
    - Residual connections
    - Weight normalization for training stability
    """

    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()

        # First causal convolution
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        # Chomp to make it causal (remove future information)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        # Second causal convolution
        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        # Combine into sequential network
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)

        # Residual connection (1x1 conv if dimensions don't match)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

        self.init_weights()

    def init_weights(self):
        """Initialize weights for better training"""
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        """
        Args:
            x: (batch, channels, time)
        Returns:
            (batch, channels, time)
        """
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class Chomp1d(nn.Module):
    """
    Removes the last n elements from the time dimension to ensure causality.
    This prevents the model from seeing future information.
    """

    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        """
        Args:
            x: (batch, channels, time)
        Returns:
            (batch, channels, time - chomp_size)
        """
        return x[:, :, :-self.chomp_size].contiguous() if self.chomp_size > 0 else x


class TemporalConvNet(nn.Module):
    """
    Temporal Convolutional Network with stacked dilated convolutions.

    The network progressively increases dilation to capture dependencies
    at different time scales, creating a large receptive field efficiently.
    """

    def __init__(self, num_inputs, num_channels, kernel_size=3, dropout=0.2):
        """
        Args:
            num_inputs: Number of input channels
            num_channels: List of channel sizes for each layer [c1, c2, c3, ...]
            kernel_size: Kernel size for temporal convolutions
            dropout: Dropout probability
        """
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)

        for i in range(num_levels):
            dilation_size = 2 ** i  # Exponentially increasing dilation
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            padding = (kernel_size - 1) * dilation_size  # Causal padding

            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1,
                                    dilation=dilation_size, padding=padding, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        Args:
            x: (batch, channels, time)
        Returns:
            (batch, channels, time)
        """
        return self.network(x)


class TCNModel(nn.Module):
    """
    TCN-based model for human motion prediction.

    Architecture:
    1. Input embedding layer
    2. Temporal Convolutional Network (encoder)
    3. Decoder with attention mechanism
    4. Output projection layer

    The TCN efficiently captures long-term temporal dependencies
    using dilated causal convolutions with a large receptive field.
    """

    def __init__(self, source_seq_len, target_seq_len, input_size,
                 hidden_size=256, num_layers=4, kernel_size=3, dropout=0.3,
                 residual_velocities=False):
        """
        Args:
            source_seq_len: Length of input sequence
            target_seq_len: Length of output prediction sequence
            input_size: Dimensionality of input features (e.g., 54)
            hidden_size: Hidden dimension for TCN layers
            num_layers: Number of TCN layers
            kernel_size: Kernel size for temporal convolutions
            dropout: Dropout probability
            residual_velocities: If True, predict velocities instead of absolute poses
        """
        super(TCNModel, self).__init__()

        self.source_seq_len = source_seq_len
        self.target_seq_len = target_seq_len
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.residual_velocities = residual_velocities

        # Input embedding
        self.input_fc = nn.Linear(input_size, hidden_size)

        # Encoder TCN
        # Create channel list with exponentially growing then constant channels
        encoder_channels = [hidden_size] * num_layers
        self.encoder_tcn = TemporalConvNet(
            num_inputs=hidden_size,
            num_channels=encoder_channels,
            kernel_size=kernel_size,
            dropout=dropout
        )

        # Decoder TCN
        decoder_channels = [hidden_size] * num_layers
        self.decoder_tcn = TemporalConvNet(
            num_inputs=hidden_size,
            num_channels=decoder_channels,
            kernel_size=kernel_size,
            dropout=dropout
        )

        # Attention mechanism for encoder-decoder
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )

        # Output projection
        self.output_fc = nn.Linear(hidden_size, input_size)

        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_size)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        self.init_weights()

    def init_weights(self):
        """Initialize weights"""
        self.input_fc.weight.data.normal_(0, 0.01)
        self.output_fc.weight.data.normal_(0, 0.01)

    def forward(self, encoder_inputs, decoder_inputs):
        """
        Forward pass with teacher forcing.

        Args:
            encoder_inputs: (batch, source_seq_len, input_size)
            decoder_inputs: (batch, target_seq_len, input_size)
        Returns:
            outputs: (batch, target_seq_len, input_size)
        """
        batch_size = encoder_inputs.size(0)

        # Encode input sequence
        # (batch, seq_len, input_size) -> (batch, seq_len, hidden_size)
        encoder_embedded = self.input_fc(encoder_inputs)
        encoder_embedded = self.dropout(encoder_embedded)

        # TCN expects (batch, channels, time)
        encoder_embedded = encoder_embedded.permute(0, 2, 1)
        encoder_features = self.encoder_tcn(encoder_embedded)
        encoder_features = encoder_features.permute(0, 2, 1)  # Back to (batch, time, channels)

        # Decode with teacher forcing
        decoder_embedded = self.input_fc(decoder_inputs)
        decoder_embedded = self.dropout(decoder_embedded)

        # TCN processing
        decoder_embedded = decoder_embedded.permute(0, 2, 1)
        decoder_features = self.decoder_tcn(decoder_embedded)
        decoder_features = decoder_features.permute(0, 2, 1)

        # Attention between decoder and encoder
        # (batch, target_seq_len, hidden_size) attending to (batch, source_seq_len, hidden_size)
        attended_features, _ = self.attention(
            decoder_features,  # query
            encoder_features,  # key
            encoder_features   # value
        )

        # Residual connection and layer norm
        decoder_output = self.layer_norm(decoder_features + attended_features)

        # Project to output space
        outputs = self.output_fc(decoder_output)  # (batch, target_seq_len, input_size)

        # Add residual velocities if enabled
        if self.residual_velocities:
            # Compute velocities and add to last known pose
            last_pose = encoder_inputs[:, -1:, :]  # (batch, 1, input_size)
            for t in range(self.target_seq_len):
                if t == 0:
                    outputs[:, t, :] = outputs[:, t, :] + last_pose.squeeze(1)
                else:
                    outputs[:, t, :] = outputs[:, t, :] + outputs[:, t-1, :]

        return outputs

    def predict(self, encoder_inputs, target_seq_len):
        """
        Autoregressive prediction for inference.

        Args:
            encoder_inputs: (batch, source_seq_len, input_size)
            target_seq_len: Number of future frames to predict
        Returns:
            predictions: (batch, target_seq_len, input_size)
        """
        batch_size = encoder_inputs.size(0)
        device = encoder_inputs.device

        # Encode input sequence
        encoder_embedded = self.input_fc(encoder_inputs)
        encoder_embedded = encoder_embedded.permute(0, 2, 1)
        encoder_features = self.encoder_tcn(encoder_embedded)
        encoder_features = encoder_features.permute(0, 2, 1)

        # Autoregressive decoding
        predictions = []
        decoder_input = encoder_inputs[:, -1:, :]  # Start with last input

        for t in range(target_seq_len):
            # Embed current input
            decoder_embedded = self.input_fc(decoder_input)
            decoder_embedded = decoder_embedded.permute(0, 2, 1)
            decoder_features = self.decoder_tcn(decoder_embedded)
            decoder_features = decoder_features.permute(0, 2, 1)

            # Attention
            attended_features, _ = self.attention(
                decoder_features,
                encoder_features,
                encoder_features
            )

            # Combine and normalize
            decoder_output = self.layer_norm(decoder_features + attended_features)

            # Project to output
            output = self.output_fc(decoder_output[:, -1, :])  # (batch, input_size)

            # Add residual if enabled
            if self.residual_velocities:
                output = output + decoder_input.squeeze(1)

            predictions.append(output)

            # Update decoder input for next step
            decoder_input = output.unsqueeze(1)

        predictions = torch.stack(predictions, dim=1)  # (batch, target_seq_len, input_size)
        return predictions


class TCNFormerModel(nn.Module):
    """
    Hybrid TCN-Transformer model combining strengths of both architectures.

    Uses TCN for efficient temporal feature extraction and Transformer
    for capturing long-range dependencies with attention.

    Based on TCNFormer (2024) research.
    """

    def __init__(self, source_seq_len, target_seq_len, input_size,
                 hidden_size=256, num_tcn_layers=3, num_transformer_layers=2,
                 num_heads=4, kernel_size=3, dropout=0.3,
                 residual_velocities=False):
        """
        Args:
            source_seq_len: Length of input sequence
            target_seq_len: Length of output prediction sequence
            input_size: Dimensionality of input features
            hidden_size: Hidden dimension
            num_tcn_layers: Number of TCN layers
            num_transformer_layers: Number of Transformer layers
            num_heads: Number of attention heads
            kernel_size: TCN kernel size
            dropout: Dropout probability
            residual_velocities: If True, predict velocities
        """
        super(TCNFormerModel, self).__init__()

        self.source_seq_len = source_seq_len
        self.target_seq_len = target_seq_len
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.residual_velocities = residual_velocities

        # Input embedding
        self.input_fc = nn.Linear(input_size, hidden_size)

        # TCN encoder for efficient feature extraction
        tcn_channels = [hidden_size] * num_tcn_layers
        self.tcn_encoder = TemporalConvNet(hidden_size, tcn_channels, kernel_size, dropout)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(hidden_size, dropout, max_len=source_seq_len + target_seq_len)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_transformer_layers)

        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_transformer_layers)

        # Output projection
        self.output_fc = nn.Linear(hidden_size, input_size)

        self.dropout = nn.Dropout(dropout)

    def forward(self, encoder_inputs, decoder_inputs):
        """
        Args:
            encoder_inputs: (batch, source_seq_len, input_size)
            decoder_inputs: (batch, target_seq_len, input_size)
        Returns:
            outputs: (batch, target_seq_len, input_size)
        """
        # Embed inputs
        encoder_embedded = self.input_fc(encoder_inputs)

        # TCN feature extraction
        encoder_embedded = encoder_embedded.permute(0, 2, 1)
        encoder_tcn_features = self.tcn_encoder(encoder_embedded)
        encoder_tcn_features = encoder_tcn_features.permute(0, 2, 1)

        # Add positional encoding
        encoder_tcn_features = self.pos_encoding(encoder_tcn_features)

        # Transformer encoding
        encoder_output = self.transformer_encoder(encoder_tcn_features)

        # Decode
        decoder_embedded = self.input_fc(decoder_inputs)
        decoder_embedded = self.pos_encoding(decoder_embedded)

        # Transformer decoding
        decoder_output = self.transformer_decoder(decoder_embedded, encoder_output)

        # Project to output
        outputs = self.output_fc(decoder_output)

        # Add residual velocities if enabled
        if self.residual_velocities:
            last_pose = encoder_inputs[:, -1:, :]
            for t in range(self.target_seq_len):
                if t == 0:
                    outputs[:, t, :] = outputs[:, t, :] + last_pose.squeeze(1)
                else:
                    outputs[:, t, :] = outputs[:, t, :] + outputs[:, t-1, :]

        return outputs

    def predict(self, encoder_inputs, target_seq_len):
        """Autoregressive prediction"""
        batch_size = encoder_inputs.size(0)
        device = encoder_inputs.device

        # Encode
        encoder_embedded = self.input_fc(encoder_inputs)
        encoder_embedded = encoder_embedded.permute(0, 2, 1)
        encoder_tcn_features = self.tcn_encoder(encoder_embedded)
        encoder_tcn_features = encoder_tcn_features.permute(0, 2, 1)
        encoder_tcn_features = self.pos_encoding(encoder_tcn_features)
        encoder_output = self.transformer_encoder(encoder_tcn_features)

        # Autoregressive decoding
        predictions = []
        decoder_input = encoder_inputs[:, -1:, :]

        for t in range(target_seq_len):
            decoder_embedded = self.input_fc(decoder_input)
            decoder_embedded = self.pos_encoding(decoder_embedded)
            decoder_output = self.transformer_decoder(decoder_embedded, encoder_output)
            output = self.output_fc(decoder_output[:, -1, :])

            if self.residual_velocities:
                output = output + decoder_input.squeeze(1)

            predictions.append(output)
            decoder_input = output.unsqueeze(1)

        predictions = torch.stack(predictions, dim=1)
        return predictions


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            (batch, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
