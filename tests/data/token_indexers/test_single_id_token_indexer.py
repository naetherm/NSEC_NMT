# -*- coding: utf-8 -*-
'''
Copyright 2019, University of Freiburg.
Chair of Algorithms and Data Structures.
Markus Näther <naetherm@informatik.uni-freiburg.de>
'''

import argparse
import unittest

from collections import defaultdict
import torch

import tests.utils as test_utils

from fairseq.data.dictionary import Dictionary
from fairseq.data.tokenizers.token import Token
from fairseq.data.token_indexers.single_id_token_indexer import SingleIdTokenIndexer

class TestSingleIdTokenIndexer(unittest.TestCase):

  def test_count_vocab_items_respect_casing(self):
    indexer = SingleIdTokenIndexer("words")
    counter = defaultdict(lambda: defaultdict(int))
    indexer.count_vocab_items(Token("Hello"), counter)
    indexer.count_vocab_items(Token("hello"), counter)

    assert counter["words"] == {"hello": 1, "Hello": 1}
  

  def test_count_vocab_items_case_insensitive(self):
    indexer = SingleIdTokenIndexer("words", lowercase_tokens=True)
    counter = defaultdict(lambda: defaultdict(int))
    indexer.count_vocab_items(Token("Hello"), counter)
    indexer.count_vocab_items(Token("hello"), counter)

    assert counter["words"] == {"hello": 2}