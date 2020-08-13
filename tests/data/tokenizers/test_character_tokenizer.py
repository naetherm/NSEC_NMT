# -*- coding: utf-8 -*-
'''
Copyright 2019, University of Freiburg.
Chair of Algorithms and Data Structures.
Markus Näther <naetherm@informatik.uni-freiburg.de>
'''

import argparse
import unittest

import torch

import tests.utils as test_utils

from fairseq.data.tokenizers.character_tokenizer import CharacterTokenizer

class TestCharacterTokenizer(unittest.TestCase):

  def test_splits_into_characters(self):
    tokenizer = CharacterTokenizer(start_tokens=['<S1>', '<S2>'], end_tokens=['</S2>', '</S1>'])
    sentence = "Small sentence."
    tokens = [t.text for t in tokenizer.tokenize(sentence)]
    expected_ = ["<S1>", "<S2>", "S", "m", "a", "l", "l", " ", "s", "e", "n", "t", "e", "n", "c", "e", ".", "</S2>", "</S1>"]
    
    assert tokens == expected_

  def test_batch_tokenization(self):
    tokenizer = CharacterTokenizer()
    sentences = [
      "Small sentence.",
      "Second sentence.",
      "Third sentence!"
    ]
    batched_tokens = tokenizer.batch_tokenize(sentences)
    single_tokens = [tokenizer.tokenize(s) for s in sentences]
    
    assert len(batched_tokens) == len(single_tokens)

    for b, s in zip(batched_tokens, single_tokens):
      assert len(b) == len(s)
      for bw, sw in zip(b, s):
        assert bw.text == sw.text
  
  def test_handles_byte_encoding(self):
    tokenizer = CharacterTokenizer(byte_encoding='utf-8', start_tokens=[259], end_tokens=[260])
    word = "åøâáabe"
    tokens = [t.text_id for t in tokenizer.tokenize(word)]
    # Note that we've added one to the utf-8 encoded bytes, to account for masking.
    expected_ = [259, 196, 166, 196, 185, 196, 163, 196, 162, 98, 99, 102, 260]
    assert tokens == expected_