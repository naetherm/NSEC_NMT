# -*- coding: utf-8 -*-
'''
Copyright 2019, University of Freiburg.
Chair of Algorithms and Data Structures.
Markus Näther <naetherm@informatik.uni-freiburg.de>
'''

from typing import Any, Dict, List, Mapping

from overrides import overrides

from fairseq.data.fields.field import Field, DataArray

class MetadataField(Field[DataArray], Mapping[str, Any]):

  def __init__(self, metadata: str) -> None:
    self.metadata = metadata

  def __getitem__(self, key: str) -> Any:
    try:
      return self.metadata[key]
    except TypeError:
      raise TypeError("Your Metadata is not a dictionary")

  def __iter__(self):
    try:
      return iter(self.metadata)
    except TypeError:
      raise TypeError("Your Metadata is not iterable.")

  def __len__(self):
    try:
      return len(self.metadata)
    except TypeError:
      raise TypeError("Your Metadata has no length")

  @overrides
  def get_padding_lengths(self) -> Dict[str, int]:
    return {}

  @overrides
  def as_tensor(self, padding_lengths: Dict[str, int]) -> DataArray:
    return self.metadata

  @overrides
  def empty_field(self) -> 'MetadataField':
    return MetadataField(None)

  @classmethod
  @overrides
  def batch_tensors(cls, tensor_list: List[DataArray]) -> List[DataArray]:
    return tensor_list

  def __str__(self) -> str:
    return f"Metadatafield"