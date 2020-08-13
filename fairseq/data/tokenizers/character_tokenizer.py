# -*- coding: utf-8 -*-
'''
Copyright 2019, University of Freiburg.
Chair of Algorithms and Data Structures.
Markus Näther <naetherm@informatik.uni-freiburg.de>
'''

from typing import List
from overrides import overrides

from fairseq.data.tokenizers.token import Token
from fairseq.data.tokenizers.tokenizer import Tokenizer

@Tokenizer.register("character")
class CharacterTokenizer(Tokenizer):

  def __init__(
    self,
    byte_encoding: str = None,
    lowercase_characters: bool = False,
    start_tokens: List[str] = None,
    end_tokens: List[str] = None):
    self.byte_encoding = byte_encoding
    self.lowercase_characters = lowercase_characters
    self.start_tokens = start_tokens or []
    self.start_tokens.reverse()
    self.end_tokens = end_tokens or []

  @overrides
  def batch_tokenize(self, texts: List[str]) -> List[List[Token]]:
    return [self.tokenize(w) for w in texts]
  
  @overrides
  def tokenize(self, text: str) -> List[Token]:
    if self.lowercase_characters:
      text = text.lower()
    if self.byte_encoding is not None:
      # We add 1 here so that we can still use 0 for masking, no matter what bytes we get out
      # of this.
      tokens = [Token(text_id=c + 1) for c in text.encode(self.byte_encoding)]
    else:
      tokens = [Token(t) for t in list(text)]
    for start_token in self.start_tokens:
      if isinstance(start_token, int):
        token = Token(text_id=start_token, idx=0)
      else:
        token = Token(text=start_token, idx=0)
      tokens.insert(0, token)
    for end_token in self.end_tokens:
      if isinstance(end_token, int):
        token = Token(text_id=end_token, idx=0)
      else:
        token = Token(text=end_token, idx=0)
      tokens.append(token)
    return tokens