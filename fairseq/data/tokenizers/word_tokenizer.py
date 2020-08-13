# -*- coding: utf-8 -*-
'''
Copyright 2019, University of Freiburg.
Chair of Algorithms and Data Structures.
Markus Näther <naetherm@informatik.uni-freiburg.de>
'''

from typing import List
from overrides import overrides

from fairseq.data.tokenizers.tokenizer import Tokenizer
from fairseq.data.tokenizers.token import Token
from fairseq.data.tokenizers.word_stemmer import WordStemmer, PassThroughWordStemmer
from fairseq.data.tokenizers.word_splitter import WordSplitter, SimpleWordSplitter
from fairseq.data.tokenizers.word_filter import WordFilter, PassThroughWordFilter

@Tokenizer.register("word")
class WordTokenizer(Tokenizer):
  """
  A :class: `WordTokenizer` handles the splitting of sentences into words. You can 
  define more post-processing steps like word filtering, word stemming, etc.
  Keep in mind that you have to specify a word_splitter, otherwise the simple :class:
  `SimpleWordSplitter` will be used.
  """

  def __init__(
    self,
    word_splitter: WordSplitter = None,
    word_filter: WordFilter = PassThroughWordFilter(),
    word_stemmer: WordStemmer = PassThroughWordStemmer(),
    start_tokens: List[str] = None,
    end_tokens: List[str] = None
  ) -> None:
    self.word_splitter = word_splitter or SimpleWordSplitter()
    self.word_filter = word_filter
    self.word_stemmer = word_stemmer
    self.start_tokens = start_tokens or []
    self.start_tokens.reverse()

    self.end_tokens = end_tokens or []

  @overrides
  def tokenize(self, text: str) -> List[Token]:
    # First split the words 
    splitted = self.word_splitter.split_words(text)

    # Second filter out words
    filtered = self.word_filter.filter(splitted)

    # Stem the words
    stemmed = self.word_stemmer.stem(filtered)

    # Finalize
    for start_token in self.start_tokens:
      stemmed.insert(0, Token(start_token, 0))
    for end_token in self.end_tokens:
      stemmed.append(Token(end_token, -1))
    
    return stemmed

  @overrides
  def batch_tokenize(self, texts: List[str]) -> List[List[Token]]:
    return [self.tokenize(s) for s in texts]