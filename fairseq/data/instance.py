# -*- coding: utf-8 -*-
'''
Copyright 2019, University of Freiburg.
Chair of Algorithms and Data Structures.
Markus Näther <naetherm@informatik.uni-freiburg.de>
'''

from typing import Dict, MutableMapping, Mapping

from fairseq.data.fields.field import Field, DataArray
from fairseq.data.dictionary import Dictionary

class Instance(Mapping[str, Field]):
  """
  An ``Instance`` is a collection of :class:`~faitseq.data.fields.field.Field` objects,
  specifying the inputs and outputs to
  some model.  We don't make a distinction between inputs and outputs here, though - all
  operations are done on all fields, and when we return arrays, we return them as dictionaries
  keyed by field name.  A model can then decide which fields it wants to use as inputs as which
  as outputs.
  The ``Fields`` in an ``Instance`` can start out either indexed or un-indexed.  During the data
  processing pipeline, all fields will be indexed, after which multiple instances can be combined
  into a ``Batch`` and then converted into padded arrays.
  """

  def __init__(
    self,
    fields: MutableMapping[str, Field]
  ) -> None:
    """
    Constructor.

    Parameters
    ----------
    fields : ``Dict[str, Field]``
        The ``Field`` objects that will be used to produce data arrays for this instance.
    """
    self.fields = fields
    self.is_indexed = False

  def __getitem__(self, key: str) -> Field:
    return self.fields[key]

  def __iter__(self):
    return iter(self.fields)

  def __len__(self) -> int:
    return len(self.fields)

  def add_field(
    self, 
    field_name: str,
    field: Field,
    vocab: Dictionary = None
  ) -> None:
    self.fields[field_name] = field
    if self.is_indexed:
      field.index(vocab)

  def count_vocab_items(
    self, 
    counter: Dict[str, Dict[str, int]]
  ):
    for f in self.fields.values():
      f.count_vocab_items(counter)

  def index_fields(self, vocab: Dictionary) -> None:
    if not self.is_indexed:
      self.is_indexed = True
      for f in self.fields.values():
        f.index(vocab)

  def get_padding_lengths(self) -> Dict[str, Dict[str, int]]:
    lengths = {}

    for n, f in self.fields.items():
      lengths[n] = f.get_padding_lengths()

    return lengths

  def as_tensor_dict(
    self,
    padding_lengths: Dict[str, Dict[str, int]] = None
  ) -> Dict[str, DataArray]:
    padding_lengths = padding_lengths or self.get_padding_lengths()
    tensors = {}

    for n, f in self.fields.items():
      tensors[n] = f.as_tensor(padding_lengths[n])

    return tensors

  def __str__(self) -> str:
    base_ = f"Instance with fields:\n"
    return " ".join([base_] + [f"\t {name}: {field} \n" for name, field in self.fields.items()])