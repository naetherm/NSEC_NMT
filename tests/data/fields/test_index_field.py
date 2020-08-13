# -*- coding: utf-8 -*-
'''
Copyright 2019, University of Freiburg.
Chair of Algorithms and Data Structures.
Markus Näther <naetherm@informatik.uni-freiburg.de>
'''

import argparse
import unittest

import numpy
import torch

import tests.utils as test_utils

from fairseq.data.tokenizers.token import Token
from fairseq.data.fields.text_field import TextField
from fairseq.data.fields.index_field import IndexField
from fairseq.data.token_indexers.single_id_token_indexer import SingleIdTokenIndexer

class TestIndexField(unittest.TestCase):

  def setUp(self):

    super(TestIndexField, self).setUp()

    self.text = TextField([Token(t) for t in ["This", "is", "a", "sentence", "."]], {"words": SingleIdTokenIndexer("words")})

  def test_as_tensor_converts_field_correctly(self):
    idx_field = IndexField(4, self.text)

    tensor = idx_field.as_tensor(idx_field.get_padding_lengths()).detach().cpu().numpy()
    numpy.testing.assert_array_equal(tensor, numpy.array([4]))

  def test_index_field_empty_field(self):
    idx_field = IndexField(4, self.text)
    empty_ = idx_field.empty_field()

    assert empty_.sequence_idx == -1

  def test_printing(self):
    print(self.text)

  def test_equality_check(self):
    idx_field1 = IndexField(4, self.text)
    idx_field2 = IndexField(4, self.text)

    assert idx_field1 == 4
    assert idx_field1 == idx_field1
    assert idx_field2 == idx_field2
    assert idx_field1 != idx_field2