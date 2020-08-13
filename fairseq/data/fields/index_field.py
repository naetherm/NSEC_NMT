# -*- coding: utf-8 -*-
'''
Copyright 2019, University of Freiburg.
Chair of Algorithms and Data Structures.
Markus Näther <naetherm@informatik.uni-freiburg.de>
'''

from typing import Dict, List
from overrides import overrides

import numpy
import torch

from fairseq.data.fields.field import Field
from fairseq.data.fields.sequence_field import SequenceField
from fairseq.data.dictionary import Dictionary

class IndexField(Field[torch.Tensor]):

  def __init__(
    self,
    index: int,
    sequence_field: SequenceField
  ) -> None:
    self.sequence_idx = index
    self.sequence_field = sequence_field

  @overrides
  def get_padding_lengths(self) -> Dict[str, int]:
    return {}

  @overrides
  def as_tensor(self, padding_length: Dict[str, int]) -> torch.Tensor:
    tensor = torch.LongTensor([self.sequence_idx])

    return tensor
    
  @overrides
  def empty_field(self) -> 'Field':
    return IndexField(-1, self.sequence_field.empty_field())

  @overrides
  def __str__(self) -> str:
    return f"IndexField [{self.sequence_idx}]."

  @overrides
  def __eq__(self, other) -> bool:
    if isinstance(other, int):
      return self.sequence_idx == other
    else:
      return id(other) == id(self)