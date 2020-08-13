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
from fairseq.data.dictionary import Dictionary

class ArrayField(Field[numpy.ndarray]):

  def __init__(
    self,
    array: numpy.ndarray,
    padding_value: int = 0,
    dtype: numpy.dtype = numpy.float32
  ) -> None:
    super(ArrayField, self).__init__()

    self.array = array
    self.padding_value = padding_value
    self.dtype = dtype
  
  @overrides
  def count_vocab_items(self, counter: Dict[str, Dict[str, int]]):
    pass

  @overrides
  def index(self, vocab: Dictionary):
    pass

  @overrides
  def get_padding_lengths(self) -> Dict[str, int]:
    return {"dimension_" + str(i): shape for i, shape in enumerate(self.array.shape)}

  @overrides
  def as_tensor(self, padding_length: Dict[str, int]) -> numpy.ndarray:
    max_shape = [padding_length["dimension_{}".format(i)] for i in range(len(padding_length))]

    # Convert explicitly to an ndarray just in case it's an scalar
    # (it'd end up not being an ndarray otherwise).
    # Also, the explicit dtype declaration for `asarray` is necessary for scalars.
    return_array = numpy.asarray(numpy.ones(max_shape, dtype=self.dtype) * self.padding_value, dtype=self.dtype)

    # If the tensor has a different shape from the largest tensor, pad dimensions with zeros to
    # form the right shaped list of slices for insertion into the final tensor.
    slicing_shape = list(self.array.shape)
    if len(self.array.shape) < len(max_shape):
      slicing_shape = slicing_shape + [0 for _ in range(len(max_shape) - len(self.array.shape))]

    slices = tuple([slice(0, x) for x in slicing_shape])
    return_array[slices] = self.array
    tensor = torch.from_numpy(return_array)
    
    return tensor
    
  @overrides
  def empty_field(self):
    return ArrayField(
      numpy.array([], dtype=self.dtype),
      padding_value=self.padding_value,
      dtype=self.dtype
    )

  def __str__(self) -> str:
    return f"ArrayField with shape: {self.array.shape} and dtype: {self.dtype}."