# -*- coding: utf-8 -*-
'''
Copyright 2019, University of Freiburg.
Chair of Algorithms and Data Structures.
Markus Näther <naetherm@informatik.uni-freiburg.de>
'''

import torch

from torch import nn
import torch.nn.functional as F

class SNU(torch.nn.Module):
  """
  """

  def __init__(
    self,
    in_features: int,
    out_features: int,
    l_tau: float = 0.8,
    soft: bool = True,
    recurrent: bool = True,
    bias: bool = True,
    initial_bias = None
  ) -> None:
    """
    Constructor.

    Parameters
    ----------
    """
    super(SNU, self).__init__()

    # Collect important variables
    self.in_features = in_features
    self.out_features = out_features
    self.l_tau = l_tau
    self.is_soft = soft
    self.is_recurrent = recurrent
    self.has_bias = bias

    # Build the SNU

    self.Wx = torch.nn.Linear(self.in_features, self.out_features, False)
    torch.nn.init.xavier_uniform_(self.Wx.weight)
    if self.is_recurrent:
      self.Wy = torch.nn.Linear(self.out_features, self.out_features, False)
      torch.nn.init.xavier_uniform_(self.Wy.weight)

    if self.has_bias:
      self.bias = torch.nn.init.constant_(torch.empty(self.out_features), 0.)#.cuda()
    else:
      self.bias = None

    self.s = None
    self.y = None

  def reset_state(self, s=None, y=None):
    self.s = s
    self.y = y

  def initialize_state(self, shape):
    self.s = torch.nn.init.normal_(torch.empty((shape[-2], self.out_features)))#.cuda()
    self.y = torch.nn.init.zeros_(torch.empty((shape[-2], self.out_features)))#.cuda()

  def forward(
    self,
    x: torch.Tensor
  ):
    """
    Forward
    """
    if self.s is None:
      self.initialize_state(x.shape)

    # If the SNU is recurrent we must act differently
    if self.is_recurrent:
      s = F.elu(self.Wx(x) + self.Wy(x) + self.l_tau * self.s * (1.0 - self.y))
    else:
      s = F.elu(self.Wx(x) + self.l_tau * self.s * (1.0 - self.y))
    
    # Add additive bias
    if self.has_bias:
      s = s + self.bias

    # Use softmax?
    if self.is_soft:
      y = F.sigmoid(s)
    else:
      y = F.relu(s)

    self.s = s
    self.y = y

    return y

def LinearSNU(in_features: int, out_features: int, l_tau: float = 0.8, soft: bool = True, bias: bool = True):
  """
  """
  return SNU(in_features, out_features, l_tau, soft, recurrent=False, bias=bias)

def RecurrentSNU(in_features: int, out_features: int, l_tau: float = 0.8, soft: bool = True, bias: bool = True):
  """
  """
  return SNU(in_features, out_features, l_tau, soft, recurrent=True, bias=bias)