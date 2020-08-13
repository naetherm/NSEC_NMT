# -*- coding: utf-8 -*-
'''
Copyright 2019, University of Freiburg.
Chair of Algorithms and Data Structures.
Markus Näther <naetherm@informatik.uni-freiburg.de>
'''

import torch

from torch import nn

class DropoutLinear(torch.nn.Module):
  """
  A specialized linear unit which does an additional dropout on the weighting values.

  This can be seen as a first step to a more brian like structure, where we have an 
  effect called 'synaptic pruning' which takes place during child evolution wihtin 
  the first months: There are many (unnecessary) synaptic connections which are prunned away.
  """

  def __init__(
    self,
    in_features: int,
    out_features: int,
    dropout: float = 0.2,
    bias: bool = True
  ) -> None:
    """
    Constructor.

    Parameters
    ----------
    in_features, `int`, required.
      The size of each input sample.
    out_features, 'int', required.
      The size of each output sample.
    dropout, `float`, required.
      The probability for a dropout.
    bias, 'bool', optional, default=True.
      If set to `False`, the layer will not learn an additive bias.
    """
    # Call the super class
    super(DropoutLinear, self).__init__()
    # Fetch parameters
    self.in_features = in_features
    self.out_features = out_features
    self.dropout = dropout
    self.bias = bias

    self.weights = torch.nn.init.xavier_uniform_(torch.empty(in_features, out_features))#.cuda()
    if self.bias:
      self.additive_bias = torch.nn.init.constant_(torch.empty(out_features), 0.)#.cuda()
    self.dropout_layer = torch.nn.Dropout(self.dropout)

  def forward(
    self,
    x: torch.Tensor
  ):
    """
    Simple forward method.
    """
    x = self.dropout_layer(x)
    x = torch.matmul(x, self.weights)
    if self.bias:
      x = x + self.additive_bias
    return x