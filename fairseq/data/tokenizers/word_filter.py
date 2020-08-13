# -*- coding: utf-8 -*-
'''
Copyright 2019, University of Freiburg.
Chair of Algorithms and Data Structures.
Markus Näther <naetherm@informatik.uni-freiburg.de>
'''

from typing import List
from overrides import overrides

import regex as re

from spacy.lang.en.stop_words import STOP_WORDS

from fairseq.common.registrable import Registrable

from fairseq.data.tokenizers.token import Token

class WordFilter(Registrable):

  default_implementation = "pass_through"
  """
  A :class: `WordFilter` removes words from a list of tokens.
  """

  def filter(self, words: List[Token]) -> List[Token]:
    raise NotImplementedError

@WordFilter.register("pass_through")
class PassThroughWordFilter(WordFilter):
  """
  :class: `PassThroughWordFilter` is a pass through implementation. No words
  will be filtered out here.
  """

  @overrides
  def filter(self, words: List[Token]) -> List[Token]:
    return words

@WordFilter.register("regex")
class RegExFilter(WordFilter):
  """
  :class: `RegExFilter` is a word filter that uses a list of regex patterns for 
  filtering out specific words.
  """

  def __init__(self, patterns: List[str]) -> None:
    self.patterns = patterns
    self.joined_patterns = re.compile("|".join(self.patterns))

  @overrides
  def filter(self, words: List[Token]) -> List[Token]:
    stops = [w for w in words if not self.joined_patterns.match(w.text)]
    return stops