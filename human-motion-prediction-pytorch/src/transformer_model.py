
"""Transformer model for human motion prediction."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

#from tensorflow.python.ops import array_ops
#from tensorflow.python.ops import variable_scope as vs

import random

import numpy as np
import os
from six.moves import xrange  # pylint: disable=redefined-builtin
import torch
from torch import nn
from torch.nn import LayerNorm
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn import TransformerDecoder, TransformerDecoderLayer
import torch.nn.functional as F
#import rnn_cell_extensions # my extensions of the tf repos
import data_utils

use_cuda=False

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.5, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)

class TransformerModel(nn.Module):
  """Transformer model for human motion prediction"""

  def __init__(self,
               architecture,
               source_seq_len,
               target_seq_len,
               hidden_dim, # hidden recurrent layer size
               num_layers,
               max_gradient_norm,
               batch_size,
               learning_rate,
               learning_rate_decay_factor,
               loss_to_use,
               number_of_actions,
               num_heads = 4,
               one_hot=True,
               residual_velocities=False,
               dropout=0.0,
               dtype=torch.float32):
    """Create the model.

    Args:
      architecture: [basic, tied] whether to tie the decoder and decoder.
      source_seq_len: lenght of the input sequence.
      target_seq_len: lenght of the target sequence.
      rnn_size: number of units in the rnn.
      num_layers: number of rnns to stack.
      max_gradient_norm: gradients will be clipped to maximally this norm.
      batch_size: the size of the batches used during training;
        the model construction is independent of batch_size, so it can be
        changed after initialization if this is convenient, e.g., for decoding.
      learning_rate: learning rate to start with.
      learning_rate_decay_factor: decay learning rate by this much when needed.
      loss_to_use: [supervised, sampling_based]. Whether to use ground truth in
        each timestep to compute the loss after decoding, or to feed back the
        prediction from the previous time-step.
      number_of_actions: number of classes we have.
      one_hot: whether to use one_hot encoding during train/test (sup models).
      residual_velocities: whether to use a residual connection that models velocities.
      dtype: the data type to use to store internal variables.
    """
    super(TransformerModel, self).__init__()

    self.HUMAN_SIZE = 54
    self.input_size = self.HUMAN_SIZE + number_of_actions if one_hot else self.HUMAN_SIZE

    print( "One hot is ", one_hot )
    print( "Input size is %d" % self.input_size )

    # Summary writers for train and test runs

    self.source_seq_len = source_seq_len
    self.target_seq_len = target_seq_len
    self.hidden_dim = hidden_dim
    self.batch_size = batch_size
    self.dropout = dropout

    # === Create the RNN that will keep the state ===
    #print('Hidden Dimension = {0}'.format( hidden_dim ))
    #self.cell = torch.nn.GRUCell(self.input_size, self.rnn_size)
#    self.cell2 = torch.nn.GRUCell(self.rnn_size, self.rnn_size)

    #self.fc1 = nn.Linear(self.rnn_size, self.input_size)

    ntoken = self.input_size
    ninp = hidden_dim

    self.pos_encoder = PositionalEncoding(ninp, dropout)
    encoder_layer = TransformerEncoderLayer(
        ninp, num_heads, hidden_dim, dropout
    )
    self.transformer_encoder = TransformerEncoder(
        encoder_layer=encoder_layer,
        num_layers=num_layers,
        norm=LayerNorm(ninp),
    )
    decoder_layer = TransformerDecoderLayer(
        ninp, num_heads, hidden_dim, dropout
    )
    self.transformer_decoder = TransformerDecoder(
        decoder_layer=decoder_layer,
        num_layers=num_layers,
        norm=LayerNorm(ninp),
    )

    # Use Linear instead of Embedding for continuous valued input
    self.encoder = nn.Linear(ntoken, ninp)
    self.project = nn.Linear(ninp, ntoken)
    self.ninp = ninp

  def _generate_square_subsequent_mask(self, sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = (
        mask.float()
        .masked_fill(mask == 0, float("-inf"))
        .masked_fill(mask == 1, float(0.0))
    )
    return mask  

  def forward(self, encoder_inputs, decoder_inputs, max_len=None):
    def loop_function(prev, i):
        return prev

    #print("encoder_inputs shape is: ", encoder_inputs.shape)
    encoder_inputs = torch.transpose(encoder_inputs, 0, 1)
    decoder_inputs = torch.transpose(decoder_inputs, 0, 1)

    max_len = decoder_inputs.shape[0]

    projected_src = self.encoder(encoder_inputs) * np.sqrt(self.ninp)
    pos_encoded_src = self.pos_encoder(projected_src)
    encoder_output = self.transformer_encoder(pos_encoded_src)
    tgt_mask = self._generate_square_subsequent_mask(decoder_inputs.shape[0]).to(
        device=decoder_inputs.device,
    )
    #print(tgt_mask.shape)
    #print(decoder_inputs.shape)
    if self.training:
        # Use last source pose as first input to decoder
        tgt = torch.cat((encoder_inputs[-1].unsqueeze(0), decoder_inputs[:-1]))
        pos_encoder_tgt = self.pos_encoder(
            self.encoder(tgt) * np.sqrt(self.ninp)
        )
        output = self.transformer_decoder(
            pos_encoder_tgt, encoder_output, tgt_mask=tgt_mask,
        )
        output = self.project(output)
    else:
        # greedy decoding
        decoder_input = torch.zeros(
            max_len, encoder_inputs.shape[1], encoder_inputs.shape[-1],
        ).type_as(encoder_inputs.data)
        next_pose = decoder_inputs[0].clone()
        for i in range(max_len):
            decoder_input[i] = next_pose
            pos_encoded_input = self.pos_encoder(
                self.encoder(decoder_input) * np.sqrt(self.ninp)
            )
            decoder_outputs = self.transformer_decoder(
                pos_encoded_input, encoder_output, tgt_mask=tgt_mask,
            )
            output = self.project(decoder_outputs)
            next_pose = output[i].clone()
            del output
        output = decoder_input
    return torch.transpose(output, 0, 1)


  def get_batch( self, data, actions ):
    """Get a random batch of data from the specified bucket, prepare for step.

    Args
      data: a list of sequences of size n-by-d to fit the model to.
      actions: a list of the actions we are using
    Returns
      The tuple (encoder_inputs, decoder_inputs, decoder_outputs);
      the constructed batches have the proper format to call step(...) later.
    """

    # Select entries at random
    all_keys    = list(data.keys())

    chosen_keys = np.random.choice( len(all_keys), self.batch_size )

    # How many frames in total do we need?
    total_frames = self.source_seq_len + self.target_seq_len

    encoder_inputs  = np.zeros((self.batch_size, self.source_seq_len-1, self.input_size), dtype=float)
    decoder_inputs  = np.zeros((self.batch_size, self.target_seq_len, self.input_size), dtype=float)
    decoder_outputs = np.zeros((self.batch_size, self.target_seq_len, self.input_size), dtype=float)

    for i in xrange( self.batch_size ):

      the_key = all_keys[ chosen_keys[i] ]

      # Get the number of frames
      n, ncol = data[ the_key ].shape

      # Sample somewherein the middle
      idx = np.random.randint( 16, n-total_frames )

      # Select the data around the sampled points
      data_sel = data[ the_key ][idx:idx+total_frames ,:]

      # Add the data
      encoder_inputs[i,:,0:self.input_size]  = data_sel[0:self.source_seq_len-1, :]
      decoder_inputs[i,:,0:self.input_size]  = data_sel[self.source_seq_len-1:self.source_seq_len+self.target_seq_len-1, :]
      decoder_outputs[i,:,0:self.input_size] = data_sel[self.source_seq_len:, 0:self.input_size]

    return encoder_inputs, decoder_inputs, decoder_outputs


  def find_indices_srnn( self, data, action ):
    """
    Find the same action indices as in SRNN.
    See https://github.com/asheshjain399/RNNexp/blob/master/structural_rnn/CRFProblems/H3.6m/processdata.py#L325
    """

    # Used a fixed dummy seed, following
    # https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/forecastTrajectories.py#L29
    SEED = 1234567890
    rng = np.random.RandomState( SEED )

    subject = 5
    subaction1 = 1
    subaction2 = 2

    T1 = data[ (subject, action, subaction1, 'even') ].shape[0]
    T2 = data[ (subject, action, subaction2, 'even') ].shape[0]
    prefix, suffix = 50, 100

    idx = []
    idx.append( rng.randint( 16,T1-prefix-suffix ))
    idx.append( rng.randint( 16,T2-prefix-suffix ))
    idx.append( rng.randint( 16,T1-prefix-suffix ))
    idx.append( rng.randint( 16,T2-prefix-suffix ))
    idx.append( rng.randint( 16,T1-prefix-suffix ))
    idx.append( rng.randint( 16,T2-prefix-suffix ))
    idx.append( rng.randint( 16,T1-prefix-suffix ))
    idx.append( rng.randint( 16,T2-prefix-suffix ))
    return idx

  def get_batch_srnn(self, data, action ):
    """
    Get a random batch of data from the specified bucket, prepare for step.

    Args
      data: dictionary with k:v, k=((subject, action, subsequence, 'even')),
        v=nxd matrix with a sequence of poses
      action: the action to load data from
    Returns
      The tuple (encoder_inputs, decoder_inputs, decoder_outputs);
      the constructed batches have the proper format to call step(...) later.
    """

    actions = ["directions", "discussion", "eating", "greeting", "phoning",
              "posing", "purchases", "sitting", "sittingdown", "smoking",
              "takingphoto", "waiting", "walking", "walkingdog", "walkingtogether"]

    if not action in actions:
      raise ValueError("Unrecognized action {0}".format(action))

    frames = {}
    frames[ action ] = self.find_indices_srnn( data, action )

    batch_size = 8 # we always evaluate 8 seeds
    subject    = 5 # we always evaluate on subject 5
    source_seq_len = self.source_seq_len
    target_seq_len = self.target_seq_len

    seeds = [( action, (i%2)+1, frames[action][i] ) for i in range(batch_size)]

    encoder_inputs  = np.zeros( (batch_size, source_seq_len-1, self.input_size), dtype=float )
    decoder_inputs  = np.zeros( (batch_size, target_seq_len, self.input_size), dtype=float )
    decoder_outputs = np.zeros( (batch_size, target_seq_len, self.input_size), dtype=float )

    # Compute the number of frames needed
    total_frames = source_seq_len + target_seq_len

    # Reproducing SRNN's sequence subsequence selection as done in
    # https://github.com/asheshjain399/RNNexp/blob/master/structural_rnn/CRFProblems/H3.6m/processdata.py#L343
    for i in xrange( batch_size ):

      _, subsequence, idx = seeds[i]
      idx = idx + 50

      data_sel = data[ (subject, action, subsequence, 'even') ]

      data_sel = data_sel[(idx-source_seq_len):(idx+target_seq_len) ,:]

      encoder_inputs[i, :, :]  = data_sel[0:source_seq_len-1, :]
      decoder_inputs[i, :, :]  = data_sel[source_seq_len-1:(source_seq_len+target_seq_len-1), :]
      decoder_outputs[i, :, :] = data_sel[source_seq_len:, :]


    return encoder_inputs, decoder_inputs, decoder_outputs
