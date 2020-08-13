# -*- coding: utf-8 -*-
'''
Copyright 2019, University of Freiburg.
Chair of Algorithms and Data Structures.
Markus Näther <naetherm@informatik.uni-freiburg.de>
'''

import torch

from fairseq.common.registrable import Registrable

class TokenEmbedder(torch.nn.Module, Registrable):
  """
  A :class: `TokenEmbedder` is a PyTorch Module that takes as input a tensor with integer 
  ids that are the output of a :class: `TokenIndexer`. This `TokenEmbedder` outputs 
  a vector for each of the input tokens. 

  The input of the TokenEmbedder has typically the shape ``(batch_size, num_tokens)`` 
  or ``(batch_size, num_tokens, num_characters)`` and the output will then have the 
  shape ``(batch_size, num_tokens, output_dim)``. 
  """

  default_implementation = "embedding"

  def get_output_dim(self) -> int:
    """
    Returns the output dimension of the embedder.
    """
    raise NotImplementedError