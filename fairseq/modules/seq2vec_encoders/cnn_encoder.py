# -*- coding: utf-8 -*-
'''
Copyright 2019, University of Freiburg.
Chair of Algorithms and Data Structures.
Markus Näther <naetherm@informatik.uni-freiburg.de>
'''

from typing import Optional, Tuple

from overrides import overrides

import torch
from torch.nn import Conv1d, Linear

from fairseq.modules.seq2vec_encoders.seq2vec_encoder import Seq2VecEncoder
from fairseq.nn.activation import Activation


@Seq2VecEncoder.register("cnn")
class CNNEncoder(Seq2VecEncoder):
  """
  A :class: ``CNNEncoder`` represents a combination of multiple convolution layers
  and a max-pooling layer. The input of this layer is
  ``(batch_size, num_tokens, input_dim)``
  while the output shape will be
  ``(batch_size, output_dim)``.

  The CNN has one convolution layer for each ngram filter size. Each of these
  convolution layers has an output vector of size num_filters. The number of
  times a convolution layer will be used is ``num_tokens - ngram_size + 1``.
  The max-pooling layer aggregates all these outputs from the convolution
  layers and outputs the maximum of all these layers.
  This operation is repeated for every ngram size and consequently the
  dimensionality of the output after max-pooling is
  ``len(ngram_filter_sizes) * num_filters``. This output can then optionally
  be projected down to a lower dimension which is specified by ``output_dim``.
  """

  def __init__(
    self,
    embedding_dim: int,
    num_filters: int,
    ngram_filter_sizes: Tuple[int, ] = (2, 3, 4, 5),
    conv_layer_activation: Activation = None,
    output_dim: Optional[int] = None
  ) -> None:
    """
    Constructor.
    """
    # First call the super class
    super(CNNEncoder, self).__init__()

    # Fill parameters
    self.embedding_dim = embedding_dim
    self.num_filters = num_filters
    self.ngram_filter_sizes = ngram_filter_sizes
    self.activation = conv_layer_activation or Activation.by_name('relu')()
    self.output_dim = output_dim

    # Create the convolution layers
    self.convolution_layers = [
      Conv1d(
        in_channels=self.embedding_dim,
        out_channels=self.num_filters,
        kernel_size=ngram_size
      ) for ngram_size in self.ngram_filter_sizes
    ]
    # Add them
    for i, cl in enumerate(self.convolution_layers):
      self.add_module('convlayer_{}'.format(i), cl)

    max_pool_out_dim = self.num_filters * len(self.ngram_filter_sizes)
    if self.output_dim:
      self.projection_layer = Linear(max_pool_out_dim, self.output_dim)
    else:
      # No projection in the end
      self.projection_layer = None
      self.output_dim = max_pool_out_dim

  @overrides
  def get_output_dim(self) -> int:
    """
    Returns the number of output dimensions.
    """
    return self.output_dim

  @overrides
  def get_input_dim(self) -> int:
    """
    Returns the number of input dimensions.
    """
    return self.embedding_dim

  @overrides
  def forward(self, tokens: torch.Tensor, mask: torch.Tensor):
    if mask is not None:
      tokens = tokens * mask.unsqueeze(-1).float()

    tokens = torch.transpose(tokens, 1, 2)

    filter_outs = []
    for i in range(self.convolution_layers):
      cl = getattr(self, 'convlayer_{}'.format(i))

      filter_outs.append(self.activation(cl(tokens)).max(dim=2)[0])

    max_out = torch.cat(filter_outs, dim=1) if len(filter_outs) > 1 else filter_outs[0]

    # Any projection available?
    if self.projection_layer:
      final = self.projection_layer(max_out)
    else:
      final = max_out

    return final