# -*- coding: utf-8 -*-
'''
Copyright 2019, University of Freiburg.
Chair of Algorithms and Data Structures.
Markus Näther <naetherm@informatik.uni-freiburg.de>
'''

from overrides import overrides

import torch

from fairseq.common.checks import ConfigurationError
from fairseq.modules.seq2vec_encoders.seq2vec_encoder import Seq2VecEncoder

@Seq2VecEncoder.register('pytorch_wrapper')
class PytorchWrapperSeq2Vec(Seq2VecEncoder):

  def __init__(self, module: torch.nn.modules.RNNBase) -> None:
    super(PytorchWrapperSeq2Vec, self).__init__(stateful=False)

    self.module = module

    try:
      if not self.module.batch_first:
        raise ConfigurationError("The encoder semantics assume batch first.")
    except AttributeError:
      pass

  @overrides
  def get_input_dim(self) -> int:
    """
    Returns the number of input dimensions.
    """
    return self.module.input_size

  @overrides
  def get_output_dim(self) -> int:
    """
    Returns the number of output dimensions.
    """
    try:
      is_bidirectional = self.module.bidirectional
    except AttributeError:
      is_bidirectional = False

    return self.module.hidden_size * (2 if is_bidirectional else 1)

  @overrides
  def forward(
    self,
    inputs: torch.Tensor,
    mask: torch.Tensor,
    hidden_state: torch.Tensor = None
  ) -> torch.Tensor:
    if mask is None:
      return self.module(inputs, hidden_state)[0][:, -1, :]

    batch_size = mask.size(0)

    _, state, restoration_indices, = \
      self.sort_and_run_forward(self.module, inputs, mask, hidden_state)

    # Deal with the fact the LSTM state is a tuple of (state, memory).
    if isinstance(state, tuple):
      state = state[0]

    num_layers_times_directions, num_valid, encoding_dim = state.size()
    # Add back invalid rows.
    if num_valid < batch_size:
      # batch size is the second dimension here, because pytorch
      # returns RNN state as a tensor of shape (num_layers * num_directions,
      # batch_size, hidden_size)
      zeros = state.new_zeros(num_layers_times_directions,
                              batch_size - num_valid,
                              encoding_dim)
      state = torch.cat([state, zeros], 1)

    # Restore the original indices and return the final state of the
    # top layer. Pytorch's recurrent layers return state in the form
    # (num_layers * num_directions, batch_size, hidden_size) regardless
    # of the 'batch_first' flag, so we transpose, extract the relevant
    # layer state (both forward and backward if using bidirectional layers)
    # and return them as a single (batch_size, self.get_output_dim()) tensor.

    # now of shape: (batch_size, num_layers * num_directions, hidden_size).
    unsorted_state = state.transpose(0, 1).index_select(0, restoration_indices)

    # Extract the last hidden vector, including both forward and backward states
    # if the cell is bidirectional. Then reshape by concatenation (in the case
    # we have bidirectional states) or just squash the 1st dimension in the non-
    # bidirectional case. Return tensor has shape (batch_size, hidden_size * num_directions).
    try:
      last_state_index = 2 if self.module.bidirectional else 1
    except AttributeError:
      last_state_index = 1
    last_layer_state = unsorted_state[:, -last_state_index:, :]
    return last_layer_state.contiguous().view([-1, self.get_output_dim()])