# -*- coding: utf-8 -*-
'''
Copyright 2019, University of Freiburg.
Chair of Algorithms and Data Structures.
Markus Näther <naetherm@informatik.uni-freiburg.de>
'''

import torch

from fairseq.common.registrable import Registrable
from fairseq.data.dictionary import Dictionary
from fairseq import utils

class Seq2SeqDecoder(torch.nn.Module, Registrable):
  """
  Base class for all sequence-to-sequence decoders.
  """
  def __init__(
    self,
    dictionary: Dictionary,
    onnx_trace: bool = False
  ) -> None:
    super(Seq2SeqDecoder, self).__init__()

    self.dictionary = dictionary
    self.onnx_trace = onnx_trace

  def forward(
    self,
    prev_output_tokens,
    encoder_out = None,
    **kwargs
  ):
    x, extra = self.extract_features(prev_output_tokens, encoder_out=encoder_out, **kwargs)
    x = self.output_layer(x)
    return x, extra

  def extract_features(self, prev_output_tokens, encoder_out=None, **kwargs):
    raise NotImplementedError

  def output_layer(self, features, **kwargs):
    raise NotImplementedError