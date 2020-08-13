# -*- coding: utf-8 -*-
'''
Copyright 2019, University of Freiburg.
Chair of Algorithms and Data Structures.
Markus Näther <naetherm@informatik.uni-freiburg.de>
'''

import torch

from fairseq.common.registrable import Registrable
from fairseq.data.dictionary import Dictionary

class Seq2SeqEncoder(torch.nn.Module, Registrable):

  def __init__(
    self,
    dictionary: Dictionary
  ) -> None:
    super(Seq2SeqEncoder, self).__init__()

    self.dictionary = dictionary

  def forward(self, src_tokens, src_lengths=None, **kwargs):
    raise NotImplementedError
