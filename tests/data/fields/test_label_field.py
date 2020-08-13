# -*- coding: utf-8 -*-
'''
Copyright 2019, University of Freiburg.
Chair of Algorithms and Data Structures.
Markus Näther <naetherm@informatik.uni-freiburg.de>
'''

from collections import defaultdict
from typing import Dict, List

import argparse
import unittest

import numpy
import torch

import tests.utils as test_utils

from fairseq.data.fields.label_field import LabelField
from fairseq.data.dictionary import Dictionary


class TestLabelField(unittest.TestCase):
  def test_as_tensor_returns_integer_tensor(self):
    label = LabelField(5, skip_indexing=True)

    tensor = label.as_tensor(label.get_padding_lengths())
    assert tensor.item() == 5

  def test_label_field_can_index_with_vocab(self):
    vocab = Dictionary()
    vocab.add_token_to_namespace("entailment", namespace="labels")
    vocab.add_token_to_namespace("contradiction", namespace="labels")
    vocab.add_token_to_namespace("neutral", namespace="labels")

    label = LabelField("entailment")
    label.index(vocab)
    tensor = label.as_tensor(label.get_padding_lengths())
    assert tensor.item() == 0

  def test_label_field_empty_field_works(self):
    label = LabelField("test")
    empty_label = label.empty_field()
    assert empty_label.label == -1

  def test_printing_doesnt_crash(self):
    label = LabelField("label", label_namespace="namespace")
    print(label)