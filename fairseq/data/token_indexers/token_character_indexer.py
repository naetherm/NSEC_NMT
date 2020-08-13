# -*- coding: utf-8 -*-
'''
Copyright 2019, University of Freiburg.
Chair of Algorithms and Data Structures.
Markus Näther <naetherm@informatik.uni-freiburg.de>
'''

from typing import Dict, List
import itertools
import warnings

from overrides import overrides

from fairseq.common.util import pad_sequence_to_length
from fairseq.data.tokenizers.token import Token
from fairseq.data.token_indexers.token_indexer import TokenIndexer
from fairseq.data.tokenizers.character_tokenizer import CharacterTokenizer
from fairseq.data.dictionary import Dictionary

@TokenIndexer.register("characters")
class TokenCharacterIndexer(TokenIndexer[List[int]]):
  """
  :class: ``TokenCharacterIndexer`` is a tokenizer especially for the representation of character
  input. Tokens are represented as a list of character indices.
  """

  def __init__(
    self,
    namespace: str = "token_characters",
    character_tokenizer: CharacterTokenizer = CharacterTokenizer(),
    start_tokens: List[str] = [],
    end_tokens: List[str] = [],
    min_padding_length: int = 0,
    token_min_padding_length: int = 0
  ) -> None:
    # Call super class
    super(TokenCharacterIndexer, self).__init__(token_min_padding_length)

    self.min_padding_length = min_padding_length
    self.namespace = namespace
    self.character_tokenizer = character_tokenizer
    self.start_tokens = [Token(t) for t in start_tokens]
    self.end_tokens = [Token(t) for t in end_tokens]
  
  @overrides
  def count_vocab_items(self, token: Token, counter: Dict[str, Dict[str, int]]):
    if token.text is None:
      pass # ERROR
    else:
      for c in self.character_tokenizer.tokenize(token.text):
        if getattr(c, 'text_id', None) is None:
          counter[self.namespace][c.text] += 1

  @overrides
  def tokens_to_indices(self, tokens: List[Token], vocabulary: Dictionary, index_name: str) -> Dict[str, List[List[int]]]:
    indices: List[List[int]] = []

    # Combine the tokens with start and end tokens
    for token in itertools.chain(self.start_tokens, tokens, self.end_tokens):
      token_indices: List[int] = []
      if token.text is None:
        pass # ERROR
      else:
        for c in self.character_tokenizer.tokenize(token.text):
          if getattr(c, 'text_id', None) is not None:
            idx = c.text_id
          else:
            idx = vocabulary.get_token_index(c.text, self.namespace)
          token_indices.append(idx)
        indices.append(token_indices)

    return {index_name: indices}

  @overrides
  def get_padding_token(self):
    return []

  @overrides
  def get_padding_lengths(self, token: List[int]) -> Dict[str, int]:
    return {'num_token_characters': max(len(token), self.min_padding_length)}

  @overrides
  def pad_token_sequence(
    self, 
    tokens: Dict[str, List[List[int]]], 
    desired_num_tokens: Dict[str, int], 
    padding_length: Dict[str, int]
  ) -> Dict[str, List[List[int]]]:
    key = list(tokens.keys())[0]

    padded_ = pad_sequence_to_length(
      tokens[key],
      desired_num_tokens[key],
      default_value=self.get_padding_token
    )

    # Pad all characters within the tokens
    desired_token_length = padding_length['num_token_characters']
    longest_token: List[int] = max(tokens[key], key=len, default=[])
    padding_value = 0

    if desired_token_length > len(longest_token):
      padded_.append([padding_value] * desired_token_length)

    padded_ = list(zip(*itertools.zip_longest(*padded_, fillvalue=padding_value)))

    if desired_token_length > len(longest_token):
      padded_.pop()

    return {key: [list(token[:desired_token_length]) for token in padded_]}