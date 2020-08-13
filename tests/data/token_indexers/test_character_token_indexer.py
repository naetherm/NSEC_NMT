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
from fairseq.data.token_indexers.token_character_indexer import TokenCharacterIndexer
from fairseq.data.tokenizers.character_tokenizer import CharacterTokenizer

class TestCharacterTokenIndexer(unittest.TestCase):

  def test_count_vocab_items_respect_casing(self):
    indexer = TokenCharacterIndexer("characters", min_padding_length=5)
    counter = defaultdict(lambda: defaultdict(int))
    indexer.count_vocab_items(Token("Hello"), counter)
    indexer.count_vocab_items(Token("hello"), counter)

    assert counter["characters"] == {"h": 1, "H": 1, "e": 2, "l": 4, "o": 2}
  

  def test_count_vocab_items_case_insensitive(self):
    indexer = TokenCharacterIndexer("characters", CharacterTokenizer(lowercase_characters=True), min_padding_length=5)
    counter = defaultdict(lambda: defaultdict(int))
    indexer.count_vocab_items(Token("Hello"), counter)
    indexer.count_vocab_items(Token("hello"), counter)

    assert counter["characters"] == {"h": 2, "e": 2, "l": 4, "o": 2}

  def test_as_array_produce_token_sequence(self):
    indexer = TokenCharacterIndexer("characters", min_padding_length=1)

    padded_tokens = indexer.pad_token_sequence(
      {'k': [[1, 2, 3, 4, 5], [1, 2, 3], [1]]}, 
      desired_num_tokens={'k': 4}, 
      padding_length={"num_token_characters": 10}
    )

    expected_ = {'k': [[1, 2, 3, 4, 5, 0, 0, 0, 0, 0],
      [1, 2, 3, 0, 0, 0, 0, 0, 0, 0],
      [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    }

    assert padded_tokens == expected_

  def test_token2indices_correct_characters(self):
    vocab = Dictionary()
    vocab.add_token_to_namespace("A", namespace='characters') # 2
    vocab.add_token_to_namespace("s", namespace='characters') # 3
    vocab.add_token_to_namespace("e", namespace='characters') # 4
    vocab.add_token_to_namespace("n", namespace='characters') # 5
    vocab.add_token_to_namespace("t", namespace='characters') # 6
    vocab.add_token_to_namespace("c", namespace='characters') # 7

    indexer = TokenCharacterIndexer("characters", min_padding_length=1)
    indices = indexer.tokens_to_indices([Token("sentential")], vocab, "char")
    
    expected_ = {"char": [[3, 4, 5, 6, 4, 5, 6, 1, 1, 1]]}

    assert indices == expected_

  def test_additional_start_and_end_tokens(self):
    vocab = Dictionary()
    vocab.add_token_to_namespace("A", namespace='characters') # 2
    vocab.add_token_to_namespace("s", namespace='characters') # 3
    vocab.add_token_to_namespace("e", namespace='characters') # 4
    vocab.add_token_to_namespace("n", namespace='characters') # 5
    vocab.add_token_to_namespace("t", namespace='characters') # 6
    vocab.add_token_to_namespace("c", namespace='characters') # 7
    vocab.add_token_to_namespace("<", namespace='characters') # 8
    vocab.add_token_to_namespace(">", namespace='characters') # 9
    vocab.add_token_to_namespace("/", namespace='characters') # 10

    indexer = TokenCharacterIndexer("characters", start_tokens=["<s>"], end_tokens=["</s>"], min_padding_length=1)
    indices = indexer.tokens_to_indices([Token("sentential")], vocab, "char")

    expected_ = {"char": [[8, 3, 9], [3, 4, 5, 6, 4, 5, 6, 1, 1, 1], [8, 10, 3, 9]]}

    assert indices == expected_

  def test_minimal_padding_length(self):
    sentence = "This is a test ."
    tokens = [Token(t) for t in sentence.split(" ")]
    vocab = Dictionary()
    vocab.add_token_to_namespace("T", namespace='characters') # 2
    vocab.add_token_to_namespace("h", namespace='characters') # 3
    vocab.add_token_to_namespace("i", namespace='characters') # 4
    vocab.add_token_to_namespace("s", namespace='characters') # 5
    vocab.add_token_to_namespace("a", namespace='characters') # 6
    vocab.add_token_to_namespace("t", namespace='characters') # 7
    vocab.add_token_to_namespace("e", namespace='characters') # 8
    vocab.add_token_to_namespace(".", namespace='characters') # 9
    vocab.add_token_to_namespace("y", namespace='characters') # 10
    vocab.add_token_to_namespace("m", namespace='characters') # 11
    vocab.add_token_to_namespace("n", namespace='characters') # 12

    indexer = TokenCharacterIndexer("characters", min_padding_length=10)
    indices = indexer.tokens_to_indices(tokens, vocab, "char")
    key_padding_lengths = "num_token_characters"
    value_padding_lengths = 0

    for token in indices["char"]:
      item = indexer.get_padding_lengths(token)
      value = item.values()
      value_padding_lengths = max(value_padding_lengths, max(value))
    padded_ = indexer.pad_token_sequence(indices, {"char": len(indices["char"])}, {key_padding_lengths: value_padding_lengths})

    expected_ = {"char": [
      [2, 3, 4, 5, 0, 0, 0, 0, 0, 0],
      [4, 5, 0, 0, 0, 0, 0, 0, 0, 0],
      [6, 0, 0, 0, 0, 0, 0, 0, 0, 0],
      [7, 8, 5, 7, 0, 0, 0, 0, 0, 0],
      [9, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ]}

    assert padded_ == expected_