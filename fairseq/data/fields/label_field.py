# -*- coding: utf-8 -*-
'''
Copyright 2019, University of Freiburg.
Chair of Algorithms and Data Structures.
Markus Näther <naetherm@informatik.uni-freiburg.de>
'''

from typing import Dict, Union, Set
from overrides import overrides
import logging

import numpy
import torch

from fairseq.data.fields.field import Field
from fairseq.data.dictionary import Dictionary

logger = logging.getLogger(__name__)

class LabelField(Field[torch.Tensor]):

  def __init__(
    self,
    label: Union[str, int],
    label_namespace: str = "labels",
    skip_indexing: bool = False
  ) -> None:
    self.label = label
    self.label_namespace = label_namespace
    self.label_id = None

    if skip_indexing:
      if not isinstance(label, int):
        pass # ERROR
      else:
        self.label_id = label
    else:
      if not isinstance(label, str):
        pass # ERROR

  @overrides
  def count_vocab_items(self, counter: Dict[str, Dict[str, int]]):
    if self.label_id is None:
      counter[self.label_namespace][self.label] += 1  # type: ignore

  @overrides
  def index(self, vocab: Dictionary):
    if self.label_id is None:
      self.label_id = vocab.get_token_index(self.label, self.label_namespace)  # type: ignore

  @overrides
  def get_padding_lengths(self) -> Dict[str, int]:  # pylint: disable=no-self-use
    return {}

  @overrides
  def as_tensor(self, padding_lengths: Dict[str, int]) -> torch.Tensor:
    # pylint: disable=unused-argument,not-callable
    tensor = torch.tensor(self.label_id, dtype=torch.long)
    return tensor

  @overrides
  def empty_field(self):
    return LabelField(-1, self.label_namespace, skip_indexing=True)

  def __str__(self) -> str:
    return f"LabelField with label: {self.label} in namespace: '{self.label_namespace}'.'"