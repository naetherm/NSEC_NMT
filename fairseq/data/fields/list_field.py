# -*- coding: utf-8 -*-
'''
Copyright 2019, University of Freiburg.
Chair of Algorithms and Data Structures.
Markus Näther <naetherm@informatik.uni-freiburg.de>
'''

from typing import Dict, List, Optional, Iterator
from overrides import overrides

import torch

from fairseq.common.util import pad_sequence_to_length
from fairseq.data.fields.field import Field, DataArray
from fairseq.data.fields.sequence_field import SequenceField
from fairseq.data.tokenizers.token import Token
from fairseq.data.token_indexers.token_indexer import TokenIndexer, TokenType
from fairseq.data.dictionary import Dictionary

class ListField(SequenceField[DataArray]):
  """
  The :class: ``ListField`` represents a list of other fields. This type can be very 
  useful of you present a list of answer options that themselves are ``TextFields``.
  """

  def __init__(
    self, 
    field_list: List[Field]
  ) -> None:
    field_set = set([field.__class__ for field in field_list])
    assert len(field_set) == 1, "ListFields must contain a single field type, found " + str(field_set)

    self.field_list: List[Field] = field_list

  @overrides
  def count_vocab_items(self, counter: Dict[str, Dict[str, int]]):
    for f in self.field_list:
      f.count_vocab_items(counter)

  def __iter__(self) -> Iterator[Field]:
    return iter(self.field_list)

  def __getitem__(self, idx: int) -> Field:
    return self.field_list[idx]

  def __len__(self) -> int:
    return len(self.field_list)

  @overrides
  def index(self, vocab: Dictionary):
    for f in self.field_list:
      f.index(vocab)

  @overrides
  def get_padding_lengths(self) -> Dict[str, int]:
    field_lengths = [field.get_padding_lengths() for field in self.field_list]
    padding_lengths = {'num_fields': len(self.field_list)}

    # We take the set of all possible padding keys for all fields, rather
    # than just a random key, because it is possible for fields to be empty
    # when we pad ListFields.
    possible_padding_keys = [key for field_length in field_lengths
                            for key in list(field_length.keys())]

    for key in set(possible_padding_keys):
      # In order to be able to nest ListFields, we need to scope the padding length keys
      # appropriately, so that nested ListFields don't all use the same "num_fields" key.  So
      # when we construct the dictionary from the list of fields, we add something to the
      # name, and we remove it when padding the list of fields.
      padding_lengths['list_' + key] = max(x[key] if key in x else 0 for x in field_lengths)

    # Set minimum padding length to handle empty list fields.
    for padding_key in padding_lengths:
      padding_lengths[padding_key] = max(padding_lengths[padding_key], 1)

    return padding_lengths

  @overrides
  def sequence_length(self) -> int:
    return len(self.field_list)
  
  @overrides
  def as_tensor(self, padding_lengths: Dict[str, int]) -> Dict[str, torch.Tensor]:
    padded_field_list = pad_sequence_to_length(self.field_list,
                                                padding_lengths['num_fields'],
                                                self.field_list[0].empty_field)
    # Here we're removing the scoping on the padding length keys that we added in
    # `get_padding_lengths`; see the note there for more detail.
    child_padding_lengths = {key.replace('list_', '', 1): value
                              for key, value in padding_lengths.items()
                              if key.startswith('list_')}
    padded_fields = [field.as_tensor(child_padding_lengths)
                      for field in padded_field_list]
    return self.field_list[0].batch_tensors(padded_fields)

  @overrides
  def empty_field(self):
    return ListField([self.field_list[0].empty_field()])

  @overrides
  def batch_tensors(self, tensor_list: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    return self.field_list[0].batch_tensors(tensor_list)