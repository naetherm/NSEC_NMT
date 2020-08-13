# -*- coding: utf-8 -*-
'''
Copyright 2019, University of Freiburg.
Chair of Algorithms and Data Structures.
Markus Näther <naetherm@informatik.uni-freiburg.de>
'''

from typing import Dict, List, TypeVar, Generic

from fairseq.common.registrable import Registrable

from fairseq.data.tokenizers.token import Token
from fairseq.data.dictionary import Dictionary

TokenType = TypeVar("TokenType", int, List[int])

class TokenIndexer(Generic[TokenType], Registrable):

  default_implementation = "single_id"

  def __init__(
    self,
    token_min_padding_length: int = 0
  ) -> None:
    self.token_min_padding_length = token_min_padding_length

  def count_vocab_items(self, token: Token, counter: Dict[str, Dict[str, int]]):
    raise NotImplementedError

  def tokens_to_indices(self, tokens: List[Token], vocabulary: Dictionary, index_name: str) -> Dict[str, List[TokenType]]:
    raise NotImplementedError

  def get_padding_token(self):
    raise NotImplementedError

  def get_padding_lengths(self, token: TokenType) -> Dict[str, int]:
    raise NotImplementedError

  def get_token_min_padding_length(self) -> int:
    return self.token_min_padding_length

  def pad_token_sequence(self, tokens: Dict[str, List[TokenType]], desired_num_tokens: Dict[str, int], padding_length: Dict[str, int]) -> Dict[str, List[TokenType]]:
    raise NotImplementedError

  def get_keys(self, idx_name: str) -> List[str]:

    return [idx_name]

  def __eq__(self, other) -> bool:
    if isinstance(self, other.__class__):
      return self.__dict__ == other.__dict__
    return NotImplemented