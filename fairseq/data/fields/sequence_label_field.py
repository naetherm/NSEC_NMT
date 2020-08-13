# -*- coding: utf-8 -*-
'''
Copyright 2019, University of Freiburg.
Chair of Algorithms and Data Structures.
Markus Näther <naetherm@informatik.uni-freiburg.de>
'''

from typing import Dict, List, Union, Set, Iterator
from overrides import overrides
import logging

import torch

import textwrap
from fairseq.common.checks import ConfigurationError
from fairseq.data.fields.field import Field
from fairseq.common.util import pad_sequence_to_length
from fairseq.data.fields.field import Field
from fairseq.data.fields.sequence_field import SequenceField
from fairseq.data.dictionary import Dictionary

logger = logging.getLogger(__name__)

class SequenceLabelField(Field[torch.Tensor]):

  def __init__(
    self,
    labels: Union[List[str], List[int]],
    sequence_field: SequenceField,
    label_namespace: str = None
  ) -> None:
    self.labels = labels
    self.sequence_field = sequence_field
    self.label_namespace = label_namespace
    self.indexed_labels = None

    if len(labels) != sequence_field.sequence_length():
        raise ConfigurationError("Label length and sequence length "
                                  "don't match: %d and %d" % (len(labels), sequence_field.sequence_length()))

    if all([isinstance(x, int) for x in labels]):
        self.indexed_labels = labels

    elif not all([isinstance(x, str) for x in labels]):
        raise ConfigurationError("SequenceLabelFields must be passed either all "
                                  "strings or all ints. Found labels {} with "
                                  "types: {}.".format(labels, [type(x) for x in labels]))

  def __iter__(self) -> Iterator[Union[str, int]]:
    return iter(self.labels)

  def __getitem__(self, idx: int) -> Union[str, int]:
    return self.labels[idx]

  def __len__(self) -> int:
    return len(self.labels)

  @overrides
  def count_vocab_items(self, counter: Dict[str, Dict[str, int]]):
    if self.indexed_labels is None:
      for l in self.labels:
        counter[self.label_namespace][l] += 1

  @overrides
  def index(self, vocab: Dictionary):
    if self.indexed_labels is None:
      self.indexed_labels = [vocab.get_token_index(l, self.label_namespace) for l in self.labels]

  @overrides
  def get_padding_lengths(self) -> Dict[str, int]:
    return {'num_tokens': self.sequence_field.sequence_length()}

  @overrides
  def as_tensor(self, padding_lengths: Dict[str, int]) -> torch.Tensor:
    desired_num_tokens = padding_lengths['num_tokens']
    padded_tags = pad_sequence_to_length(self.indexed_labels, desired_num_tokens)
    tensor_ = torch.LongTensor(padded_tags)
    return tensor_

  @overrides
  def empty_field(self) -> 'SequenceField':
    empty_: List[str] = []
    sequence_label_field = SequenceLabelField(empty_, self.sequence_field.empty_field())
    sequence_label_field.indexed_labels = empty_

    return sequence_label_field

  def __str__(self) -> str:
    length = self.sequence_field.sequence_length()
    formatted_labels = "".join(["\t\t" + labels + "\n" for labels in textwrap.wrap(repr(self.labels), 100)])

    return f"SequenceLabelField of length {length} with " f"labels:\n {formatted_labels} \t\tin namespace: '{self.label_namespace}'."