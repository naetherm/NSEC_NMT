# -*- coding: utf-8 -*-
'''
Copyright 2019, University of Freiburg.
Chair of Algorithms and Data Structures.
Markus Näther <naetherm@informatik.uni-freiburg.de>
'''

import torch

from fairseq.common.registrable import Registrable

class Seq2VecEncoder(torch.nn.Module, Registrable):
  """
  The :class: ``Seq2VecEncoder`` is the foundation of modules that take a sequence
  of vectors and return a single vector.

  Input shape: ``(batch_size, sequence_length, input_dim)``
  Output shape: ``(batch_size, output_dim)``
  """

  def get_input_dim(self) -> int:
    """
    Returns the number of input dimensions.
    """
    raise NotImplementedError

  def get_output_dim(self) -> int:
    """
    Returns the number of output dimensions.
    """
    raise NotImplementedError