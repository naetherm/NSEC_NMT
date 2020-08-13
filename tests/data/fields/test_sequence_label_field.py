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

from fairseq.data.fields import SequenceLabelField, TextField
from fairseq.data.dictionary import Dictionary
from fairseq.data.tokenizers.token import Token
from fairseq.data.token_indexers.single_id_token_indexer import SingleIdTokenIndexer


class TestSequenceLabelField(unittest.TestCase):

  def setUp(self):
    super(TestSequenceLabelField, self).setUp()

    self.text = TextField([Token(t) for t in ["here", "are", "some", "words", "."]], {"words": SingleIdTokenIndexer("words")})

  def test_count_vocab_items_correctly_indexes_tags(self):
    tags = ["B", "I", "O", "O", "O"]
    sequence_label_field = SequenceLabelField(tags, self.text, label_namespace="labels")

    counter = defaultdict(lambda: defaultdict(int))
    sequence_label_field.count_vocab_items(counter)

    assert counter["labels"]["B"] == 1
    assert counter["labels"]["I"] == 1
    assert counter["labels"]["O"] == 3
    assert set(counter.keys()) == {"labels"}

  def test_index_converts_field_correctly(self):
    vocab = Dictionary()
    b_index = vocab.add_token_to_namespace("B", namespace='*labels')
    i_index = vocab.add_token_to_namespace("I", namespace='*labels')
    o_index = vocab.add_token_to_namespace("O", namespace='*labels')

    tags = ["B", "I", "O", "O", "O"]
    sequence_label_field = SequenceLabelField(tags, self.text, label_namespace="*labels")
    sequence_label_field.index(vocab)

    # pylint: disable=protected-access
    assert sequence_label_field.indexed_labels == [b_index, i_index, o_index, o_index, o_index]
    # pylint: enable=protected-access

  def test_as_tensor_produces_integer_targets(self):
    vocab = Dictionary()
    vocab.add_token_to_namespace("B", namespace='*labels')
    vocab.add_token_to_namespace("I", namespace='*labels')
    vocab.add_token_to_namespace("O", namespace='*labels')

    tags = ["B", "I", "O", "O", "O"]
    sequence_label_field = SequenceLabelField(tags, self.text, label_namespace="*labels")
    sequence_label_field.index(vocab)
    padding_lengths = sequence_label_field.get_padding_lengths()
    tensor = sequence_label_field.as_tensor(padding_lengths).detach().cpu().numpy()
    numpy.testing.assert_array_almost_equal(tensor, numpy.array([0, 1, 2, 2, 2]))

  def test_printing_doesnt_crash(self):
    tags = ["B", "I", "O", "O", "O"]
    sequence_label_field = SequenceLabelField(tags, self.text, label_namespace="labels")
    print(sequence_label_field)

  def test_sequence_methods(self):
    tags = ["B", "I", "O", "O", "O"]
    sequence_label_field = SequenceLabelField(tags, self.text, label_namespace="labels")

    assert len(sequence_label_field) == 5
    assert sequence_label_field[1] == "I"
    assert [label for label in sequence_label_field] == tags