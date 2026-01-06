"""
Graph Convolutional Network (GCN) model for human motion prediction.

Based on recent research (2024):
- Multiscale Spatio-Temporal Graph Neural Networks (MST-GNN)
- Explicitly models skeleton structure as a graph
- Captures spatial dependencies between joints through graph convolutions
- Temporal dependencies through sequence processing

Reference: https://arxiv.org/abs/2108.11244
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GraphConvolution(nn.Module):
    """
    Simple GCN layer for skeleton-based motion modeling.
    Performs graph convolution: H' = D^-1 A H W
    where A is adjacency matrix, H is input features, W is learnable weights
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x, adj):
        """
        Args:
            x: (batch, num_nodes, in_features)
            adj: (num_nodes, num_nodes) adjacency matrix
        Returns:
            (batch, num_nodes, out_features)
        """
        # x: (batch, nodes, in_features)
        support = torch.matmul(x, self.weight)  # (batch, nodes, out_features)
        output = torch.matmul(adj, support)  # (batch, nodes, out_features)
        if self.bias is not None:
            output = output + self.bias
        return output


class SpatialGraphConv(nn.Module):
    """
    Spatial graph convolution block with residual connections.
    Models spatial dependencies between skeleton joints.
    """

    def __init__(self, in_channels, out_channels, dropout=0.0):
        super(SpatialGraphConv, self).__init__()
        self.gcn = GraphConvolution(in_channels, out_channels)
        self.bn = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)

        # Residual connection
        if in_channels != out_channels:
            self.residual = nn.Linear(in_channels, out_channels)
        else:
            self.residual = nn.Identity()

    def forward(self, x, adj):
        """
        Args:
            x: (batch, num_nodes, in_channels)
            adj: (num_nodes, num_nodes)
        Returns:
            (batch, num_nodes, out_channels)
        """
        identity = self.residual(x)

        out = self.gcn(x, adj)
        # Reshape for batch norm: (batch, nodes, channels) -> (batch, channels, nodes)
        out = out.permute(0, 2, 1)
        out = self.bn(out)
        out = out.permute(0, 2, 1)  # Back to (batch, nodes, channels)
        out = F.relu(out)
        out = self.dropout(out)

        return out + identity


class TemporalConv(nn.Module):
    """
    Temporal convolution for sequence modeling.
    Captures temporal dependencies across frames.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, dropout=0.0):
        super(TemporalConv, self).__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)

        if in_channels != out_channels:
            self.residual = nn.Conv1d(in_channels, out_channels, 1)
        else:
            self.residual = nn.Identity()

    def forward(self, x):
        """
        Args:
            x: (batch, channels, time)
        Returns:
            (batch, channels, time)
        """
        identity = self.residual(x)

        out = self.conv(x)
        out = self.bn(out)
        out = F.relu(out)
        out = self.dropout(out)

        return out + identity


class GCNModel(nn.Module):
    """
    Graph Convolutional Network for human motion prediction.

    Architecture:
    1. Input embedding layer
    2. Multiple spatial-temporal graph convolution blocks
    3. GRU for temporal aggregation
    4. Output projection layer

    The model explicitly uses skeleton structure via adjacency matrix.
    """

    def __init__(self, source_seq_len, target_seq_len, input_size,
                 hidden_size=256, num_layers=4, dropout=0.3,
                 residual_velocities=False):
        """
        Args:
            source_seq_len: Length of input sequence
            target_seq_len: Length of output prediction sequence
            input_size: Dimensionality of input features (e.g., 54 for joints)
            hidden_size: Hidden dimension for graph convolutions
            num_layers: Number of graph convolution layers
            dropout: Dropout probability
            residual_velocities: If True, predict velocities instead of poses
        """
        super(GCNModel, self).__init__()

        self.source_seq_len = source_seq_len
        self.target_seq_len = target_seq_len
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.residual_velocities = residual_velocities

        # Human3.6M skeleton structure (32 joints)
        # But we work with reduced representation (18 joints for 54-dim input)
        # 54 dims = 18 joints * 3 (exponential map representation)
        self.num_joints = input_size // 3

        # Build adjacency matrix for skeleton graph
        self.adj_matrix = self._build_adjacency_matrix()

        # Input embedding: project each joint's features
        self.input_embed = nn.Linear(3, hidden_size)

        # Spatial-temporal graph convolution blocks
        self.st_gcn_blocks = nn.ModuleList([
            nn.ModuleDict({
                'spatial': SpatialGraphConv(hidden_size, hidden_size, dropout),
                'temporal': TemporalConv(hidden_size * self.num_joints,
                                        hidden_size * self.num_joints,
                                        kernel_size=3, dropout=dropout)
            })
            for _ in range(num_layers)
        ])

        # Temporal aggregation with GRU
        self.gru = nn.GRU(hidden_size * self.num_joints,
                         hidden_size * self.num_joints,
                         batch_first=True)

        # Output projection: project back to joint space
        self.output_proj = nn.Linear(hidden_size, 3)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def _build_adjacency_matrix(self):
        """
        Build adjacency matrix for Human3.6M skeleton.
        Simplified structure for 18 joints.

        Joint connections based on human skeleton:
        0: Hip (root)
        1-5: Right leg chain
        6-10: Left leg chain
        11-14: Spine to head
        15-17: Right arm
        18-20: Left arm (if num_joints > 18)

        Returns normalized adjacency matrix with self-loops
        """
        num_joints = self.num_joints
        adj = np.zeros((num_joints, num_joints))

        # Add self-loops
        adj += np.eye(num_joints)

        # Define skeleton connectivity (simplified H3.6M structure)
        # These are common joint connections for human skeleton
        if num_joints >= 18:
            # Hip connections
            adj[0, 1] = adj[1, 0] = 1  # Hip to right hip
            adj[0, 6] = adj[6, 0] = 1  # Hip to left hip
            adj[0, 11] = adj[11, 0] = 1  # Hip to spine

            # Right leg
            for i in range(1, 5):
                adj[i, i+1] = adj[i+1, i] = 1

            # Left leg
            for i in range(6, 10):
                adj[i, i+1] = adj[i+1, i] = 1

            # Spine to head
            for i in range(11, 14):
                adj[i, i+1] = adj[i+1, i] = 1

            # Shoulders
            if num_joints >= 15:
                adj[13, 15] = adj[15, 13] = 1  # Neck to right shoulder
            if num_joints >= 16:
                adj[15, 16] = adj[16, 15] = 1  # Right shoulder chain
            if num_joints >= 17:
                adj[16, 17] = adj[17, 16] = 1

        # Normalize adjacency matrix: D^-1 * A
        degree = np.sum(adj, axis=1)
        degree_inv = np.power(degree, -1)
        degree_inv[np.isinf(degree_inv)] = 0
        adj_normalized = np.diag(degree_inv) @ adj

        return torch.FloatTensor(adj_normalized)

    def forward(self, encoder_inputs, decoder_inputs):
        """
        Forward pass for training (teacher forcing).

        Args:
            encoder_inputs: (batch, source_seq_len, input_size)
            decoder_inputs: (batch, target_seq_len, input_size)
        Returns:
            outputs: (batch, target_seq_len, input_size)
        """
        batch_size = encoder_inputs.size(0)
        device = encoder_inputs.device

        # Move adjacency matrix to device
        if self.adj_matrix.device != device:
            self.adj_matrix = self.adj_matrix.to(device)

        # Encode input sequence
        encoder_features = self._encode_sequence(encoder_inputs)  # (batch, hidden)

        # Decode output sequence
        outputs = []
        hidden = encoder_features.unsqueeze(0)  # (1, batch, hidden)

        for t in range(self.target_seq_len):
            # Use teacher forcing: take from decoder_inputs
            decoder_input = decoder_inputs[:, t:t+1, :]  # (batch, 1, input_size)

            # Process through spatial-temporal blocks
            dec_features = self._encode_sequence(decoder_input)  # (batch, hidden)

            # GRU step
            dec_features = dec_features.unsqueeze(1)  # (batch, 1, hidden)
            output, hidden = self.gru(dec_features, hidden)

            # Project to output space
            output = output.squeeze(1)  # (batch, hidden)
            output = output.view(batch_size, self.num_joints, self.hidden_size)
            output = self.output_proj(output)  # (batch, num_joints, 3)
            output = output.view(batch_size, self.input_size)

            outputs.append(output)

        outputs = torch.stack(outputs, dim=1)  # (batch, target_seq_len, input_size)

        # Add residual velocities if enabled
        if self.residual_velocities:
            # Add last encoder input to predictions
            last_input = encoder_inputs[:, -1:, :]  # (batch, 1, input_size)
            for t in range(self.target_seq_len):
                outputs[:, t, :] = outputs[:, t, :] + last_input.squeeze(1)
                last_input = outputs[:, t:t+1, :]

        return outputs

    def _encode_sequence(self, sequence):
        """
        Encode a sequence using spatial-temporal graph convolutions.

        Args:
            sequence: (batch, seq_len, input_size)
        Returns:
            features: (batch, hidden_size * num_joints)
        """
        batch_size, seq_len, _ = sequence.size()

        # Reshape to separate joints: (batch, seq_len, num_joints, 3)
        x = sequence.view(batch_size, seq_len, self.num_joints, 3)

        # Embed each joint's features
        x = self.input_embed(x)  # (batch, seq_len, num_joints, hidden_size)

        # Process through spatial-temporal blocks
        for block in self.st_gcn_blocks:
            # Spatial graph convolution (per frame)
            spatial_out = []
            for t in range(seq_len):
                frame = x[:, t, :, :]  # (batch, num_joints, hidden_size)
                frame = block['spatial'](frame, self.adj_matrix)
                spatial_out.append(frame)
            x = torch.stack(spatial_out, dim=1)  # (batch, seq_len, num_joints, hidden_size)

            # Temporal convolution (across frames)
            # Reshape: (batch, seq_len, num_joints, hidden) -> (batch, num_joints*hidden, seq_len)
            x_temp = x.permute(0, 2, 3, 1).contiguous()
            x_temp = x_temp.view(batch_size, self.num_joints * self.hidden_size, seq_len)
            x_temp = block['temporal'](x_temp)
            # Reshape back: (batch, num_joints*hidden, seq_len) -> (batch, seq_len, num_joints, hidden)
            x_temp = x_temp.view(batch_size, self.num_joints, self.hidden_size, seq_len)
            x = x_temp.permute(0, 3, 1, 2).contiguous()

        # Aggregate temporal information (take last frame)
        x = x[:, -1, :, :]  # (batch, num_joints, hidden_size)

        # Flatten for GRU
        x = x.view(batch_size, self.num_joints * self.hidden_size)

        return x

    def predict(self, encoder_inputs, target_seq_len):
        """
        Autoregressive prediction (for inference).

        Args:
            encoder_inputs: (batch, source_seq_len, input_size)
            target_seq_len: Number of future frames to predict
        Returns:
            predictions: (batch, target_seq_len, input_size)
        """
        batch_size = encoder_inputs.size(0)
        device = encoder_inputs.device

        # Move adjacency matrix to device
        if self.adj_matrix.device != device:
            self.adj_matrix = self.adj_matrix.to(device)

        # Encode input sequence
        encoder_features = self._encode_sequence(encoder_inputs)

        # Autoregressive decoding
        predictions = []
        hidden = encoder_features.unsqueeze(0)
        decoder_input = encoder_inputs[:, -1:, :]  # Start with last encoder input

        for t in range(target_seq_len):
            # Process current input
            dec_features = self._encode_sequence(decoder_input)

            # GRU step
            dec_features = dec_features.unsqueeze(1)
            output, hidden = self.gru(dec_features, hidden)

            # Project to output space
            output = output.squeeze(1)
            output = output.view(batch_size, self.num_joints, self.hidden_size)
            output = self.output_proj(output)
            output = output.view(batch_size, self.input_size)

            # Add residual if enabled
            if self.residual_velocities:
                output = output + decoder_input.squeeze(1)

            predictions.append(output)

            # Use prediction as next input
            decoder_input = output.unsqueeze(1)

        predictions = torch.stack(predictions, dim=1)
        return predictions
