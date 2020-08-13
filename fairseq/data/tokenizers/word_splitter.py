# -*- coding: utf-8 -*-
'''
Copyright 2019, University of Freiburg.
Chair of Algorithms and Data Structures.
Markus Näther <naetherm@informatik.uni-freiburg.de>
'''

from typing import List, Optional

from overrides import overrides
import regex as re
import spacy
import ftfy

from fairseq.common.registrable import Registrable
from fairseq.common.util import get_spacy_model
from fairseq.data.tokenizers.token import Token

class WordSplitter(Registrable):
  """
  A :class: `WordSplitter` simply splits a string into words, also called Tokenizer.
  """

  default_implementation = "simple"

  def split_words(self, sentence: str) -> List[Token]:
    raise NotImplementedError

  def batch_split_words(self, sentences: List[str]) -> List[List[Token]]:
    return [self.split_words(s) for s in sentences]

@WordSplitter.register("simple")
class SimpleWordSplitter(WordSplitter):
  """
  :class: `SimpleWordSplitter` is a very simple word splitter that splits string 
  at space positions.
  """

  @overrides
  def split_words(self, sentence: str) -> List[Token]:
    return [Token(t) for t in sentence.split()]

@WordSplitter.register("nsec")
class NSECWordSplitter(WordSplitter):
  """
  :class: `NSECWordSplitter` is our very specialized version of word and token splitting.
  """

  @overrides
  def split_words(self, sentence: str) -> List[Token]:
    tokens = re.findall(r"(?:\d+,\d+)|(?:[\w'\u0080-\u9999]+(?:[-]+[\w'\u0080-\u9999]+)+)|(?:[\w\u0080-\u9999]+(?:[']+[\w\u0080-\u9999]+)+)|\b[_]|(?:[_]*[\w\u0080-\u9999]+(?=_\b))|(?:[\w\u0080-\u9999]+)|[^\w\s\p{Z}]", sentence, re.UNICODE)

    return [Token(t) for t in tokens]