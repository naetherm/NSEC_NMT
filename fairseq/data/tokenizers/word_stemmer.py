# -*- coding: utf-8 -*-
'''
Copyright 2019, University of Freiburg.
Chair of Algorithms and Data Structures.
Markus Näther <naetherm@informatik.uni-freiburg.de>
'''

from typing import List
from overrides import overrides

from nltk.stem import PorterStemmer as NltkStemmer

from fairseq.common.registrable import Registrable

from fairseq.data.tokenizers.token import Token

class WordStemmer(Registrable):
  """
  A :class: `WordStemmer` lemmatizes words. Thereby all words are mapped to their root
  form.
  """

  default_implementation = "pass_through"

  def stem(self, word: Token) -> Token:
    raise NotImplementedError

  def batch_stem(self, words: List[Token]) -> List[Token]:
    raise NotImplementedError

@WordStemmer.register("pass_through")
class PassThroughWordStemmer(WordStemmer):
  """
  :class: `PassThroughWordStemmer` is a pass through implementation, no lemmatization is
  performed and the word is returned as it is.
  """
  @overrides
  def stem(self, word: Token) -> Token:
    return word

  @overrides
  def batch_stem(self, words: List[Token]) -> List[Token]:
    return words

@WordStemmer.register("porter")
class PorterWordStemmer(WordStemmer):
  """
  :class: `PorterWordStemmer` is a special implementation using the nltk porter stemmer 
  in the background.
  """

  def __init__(self):
    self.stemmer = NltkStemmer()

  @overrides
  def stem(self, word: Token) -> Token:
    text = self.stemmer.stem(word.text)

    return Token(
      text=text,
      idx=word.idx,
      lemma_=word.lemma_,
      pos_=word.pos_,
      tag_=word.tag_,
      dep_=word.dep_,
      ent_type_=word.ent_type_,
      text_id=getattr(word, 'text_id', None)
    )

  @overrides
  def batch_stem(self, words: List[Token]) -> List[Token]:
    return [self.stem(w) for w in words]
