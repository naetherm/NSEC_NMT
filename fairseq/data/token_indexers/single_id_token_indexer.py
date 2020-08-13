# -*- coding: utf-8 -*-
'''
Copyright 2019, University of Freiburg.
Chair of Algorithms and Data Structures.
Markus Näther <naetherm@informatik.uni-freiburg.de>
'''

from typing import List, Dict
from overrides import overrides
import itertools

from fairseq.common.util import pad_sequence_to_length
from fairseq.data.dictionary import Dictionary
from fairseq.data.tokenizers.token import Token
from fairseq.data.token_indexers.token_indexer import TokenIndexer

@TokenIndexer.register("single_id")
class SingleIdTokenIndexer(TokenIndexer[int]):
  """
  This :class:`SingleIdTokenIndexer` represents tokens as a single integer. This is usually the case
  for the usage with GloVe were the token itself is mapped to a unique index position from 
  which we then read the embedded vector representation.

  Parameters
  ----------
  namespace: `str`, optional
    We will use this namespace in the :class:`Vocabulary` to map strings to indices.
  lowercase_tokens: `bool`
    If `True` we will call *.lower() on the token before receiving the index of the token.
  start_tokens: `List[str]`
    These are prepended to the tokens provided to ``tokens_to_indices``.
  end_tokens: `List[str]`
    These are appended to the tokens provided to ``tokens_to_indices``.

  See :class: `TokenIndexer`
  """

  def __init__(
    self,
    namespace = 'tokens',
    lowercase_tokens: bool = False,
    start_tokens: List[str] = None,
    end_tokens: List[str] = None,
    token_min_padding_length: int = 0):
    super(SingleIdTokenIndexer, self).__init__(token_min_padding_length)
    self.namespace = namespace
    self.lowercase_tokens = lowercase_tokens

    self.start_tokens = [Token(t) for t in (start_tokens or [])]
    self.end_tokens = [Token(t) for t in (end_tokens or [])]

  @overrides
  def count_vocab_items(self, token: Token, counter: Dict[str, Dict[str, int]]):
    if getattr(token, 'text_id', None) is None:
      text = token.text
      if self.lowercase_tokens:
        text = text.lower()
      counter[self.namespace][text] += 1

  @overrides
  def tokens_to_indices(self, tokens: List[Token], vocabulary: Dictionary, index_name: str):
    indices: List[int] = []

    for token in itertools.chain(self.start_tokens, tokens, self.end_tokens):
      if getattr(token, 'text_id', None) is not None:
        # `text_id` being set on the token means that we aren't using the vocab, we just use
        # this id instead.
        indices.append(token.text_id)
      else:
        text = token.text
        if self.lowercase_tokens:
          text = text.lower()
        indices.append(vocabulary.get_token_index(text, self.namespace))

    return {index_name: indices}

  @overrides
  def get_padding_token(self):
    return 0

  @overrides
  def get_padding_lengths(self, token: int) -> Dict[str, int]:
    return {}

  @overrides
  def pad_token_sequence(self, tokens: Dict[str, List[int]], desired_num_tokens: Dict[str, int], padding_length: Dict[str, int]) -> Dict[str, List[int]]:
    return {key: pad_sequence_to_length(val, desired_num_tokens[key]) for key, val in tokens.items()}