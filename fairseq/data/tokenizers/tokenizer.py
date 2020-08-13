# -*- coding: utf-8 -*-
'''
Copyright 2019, University of Freiburg.
Chair of Algorithms and Data Structures.
Markus Näther <naetherm@informatik.uni-freiburg.de>
'''

from typing import List

from fairseq.common.registrable import Registrable

from fairseq.data.tokenizers.token import Token

class Tokenizer(Registrable):

  default_implementation = "word"
  """
  A ``Tokenizer`` splits strings of text into tokens.  Typically, this either splits text into
  word tokens or character tokens, and those are the two tokenizer subclasses we have implemented
  here, though you could imagine wanting to do other kinds of tokenization for structured or
  other inputs.
  As part of tokenization, concrete implementations of this API will also handle stemming,
  stopword filtering, adding start and end tokens, or other kinds of things you might want to do
  to your tokens.  See the parameters to, e.g., :class:`~.WordTokenizer`, or whichever tokenizer
  you want to use.
  If the base input to your model is words, you should use a :class:`~.WordTokenizer`, even if
  you also want to have a character-level encoder to get an additional vector for each word
  token.  Splitting word tokens into character arrays is handled separately, in the
  :class:`..token_representations.TokenRepresentation` class.
  """

  def batch_tokenize(self, text: List[str]) -> List[List[Token]]:
    raise NotImplementedError

  def tokenize(self, text: str) -> List[Token]:
    raise NotImplementedError

  def encode(self, text: str) -> List[Token]:
    """
    Added this method to be in sync with the fairseq implementation.
    """
    return self.tokenize(text)
  
  def decode(self, text: str) -> str:
    """
    Added this method to be in sync with the fairseq implementation.
    """
    return text