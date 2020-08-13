# -*- coding: utf-8 -*-
'''
Copyright 2019, University of Freiburg.
Chair of Algorithms and Data Structures.
Markus Näther <naetherm@informatik.uni-freiburg.de>
'''

from typing import Dict, List, Optional, Iterator
from overrides import overrides

import textwrap
import torch

from spacy.tokens import Token as SpacyToken

from fairseq.data.fields.sequence_field import SequenceField
from fairseq.data.tokenizers.token import Token
from fairseq.data.token_indexers.token_indexer import TokenIndexer, TokenType
from fairseq.data.dictionary import Dictionary
from fairseq.nn.util import batch_tensor_dicts

TokenList = List[TokenType]

class TextField(SequenceField[Dict[str, torch.Tensor]]):
  """
  :class: ``TextField`` ...
  """

  def __init__(self, tokens: List[Token], token_indexers: Dict[str, TokenIndexer]) -> None:
    self.tokens = tokens
    self.token_indexers = token_indexers

  def __iter__(self) -> Iterator[Token]:
    return iter(self.tokens)

  def __getitem__(self, idx: int) -> Token:
    return self.tokens[idx]

  def __len__(self) -> int:
    return len(self.tokens)

  @overrides
  def count_vocab_items(self, counter: Dict[str, Dict[str, int]]):
    for indexer in self.token_indexers.values():
      for token in self.tokens:
        indexer.count_vocab_items(token, counter)

  @overrides
  def index(self, vocab: Dictionary):
    token_arrays: Dict[str, TokenList] = {}
    indexer_name_to_indexed_token: Dict[str, List[str]] = {}
    token_index_to_indexer_name: Dict[str, str] = {}
    for indexer_name, indexer in self.token_indexers.items():
        token_indices = indexer.tokens_to_indices(self.tokens, vocab, indexer_name)
        token_arrays.update(token_indices)
        indexer_name_to_indexed_token[indexer_name] = list(token_indices.keys())
        for token_index in token_indices:
            token_index_to_indexer_name[token_index] = indexer_name
    self._indexed_tokens = token_arrays
    self._indexer_name_to_indexed_token = indexer_name_to_indexed_token
    self._token_index_to_indexer_name = token_index_to_indexer_name

  @overrides
  def get_padding_lengths(self) -> Dict[str, int]:
    lengths = []

    # Each indexer can return a different sequence length, and for indexers that return
    # multiple arrays each can have a different length.  We'll keep track of them here.
    for indexer_name, indexer in self.token_indexers.items():
      indexer_lengths = {}

      for indexed_tokens_key in self._indexer_name_to_indexed_token[indexer_name]:
        # This is a list of dicts, one for each token in the field.
        token_lengths = [indexer.get_padding_lengths(token) for token in self._indexed_tokens[indexed_tokens_key]]
        if not token_lengths:
          # This is a padding edge case and occurs when we want to pad a ListField of
          # TextFields. In order to pad the list field, we need to be able to have an
          # _empty_ TextField, but if this is the case, token_lengths will be an empty
          # list, so we add the padding for a token of length 0 to the list instead.
          token_lengths = [indexer.get_padding_lengths([])]
        # Iterate over the keys and find the maximum token length.
        # It's fine to iterate over the keys of the first token since all tokens have the same keys.
        for key in token_lengths[0]:
          indexer_lengths[key] = max(x[key] if key in x else 0 for x in token_lengths)
      lengths.append(indexer_lengths)

    padding_lengths = {}
    num_tokens = set()
    for token_index, token_list in self._indexed_tokens.items():
      indexer_name = self._token_index_to_indexer_name[token_index]
      indexer = self.token_indexers[indexer_name]
      padding_lengths[f"{token_index}_length"] = max(len(token_list), indexer.get_token_min_padding_length())
      num_tokens.add(len(token_list))

    # We don't actually use this for padding anywhere, but we used to.  We add this key back in
    # so that older configs still work if they sorted by this key in a BucketIterator.  Taking
    # the max of all of these should be fine for that purpose.
    padding_lengths['num_tokens'] = max(num_tokens)

    # Get all keys which have been used for padding for each indexer and take the max if there are duplicates.
    padding_keys = {key for d in lengths for key in d.keys()}
    for padding_key in padding_keys:
      padding_lengths[padding_key] = max(x[padding_key] if padding_key in x else 0 for x in lengths)
    return padding_lengths

  @overrides
  def sequence_length(self) -> int:
    return len(self.tokens)
  
  @overrides
  def as_tensor(self, padding_lengths: Dict[str, int]) -> Dict[str, torch.Tensor]:
    tensors = {}
    for indexer_name, indexer in self.token_indexers.items():
      desired_num_tokens = {indexed_tokens_key: padding_lengths[f"{indexed_tokens_key}_length"]
                            for indexed_tokens_key in self._indexer_name_to_indexed_token[indexer_name]}
      indices_to_pad = {indexed_tokens_key: self._indexed_tokens[indexed_tokens_key]
                        for indexed_tokens_key in self._indexer_name_to_indexed_token[indexer_name]}
      padded_array = indexer.pad_token_sequence(indices_to_pad,
                                                desired_num_tokens, padding_lengths)
      # We use the key of the indexer to recognise what the tensor corresponds to within the
      # field (i.e. the result of word indexing, or the result of character indexing, for
      # example).
      # TODO(mattg): we might someday have a TokenIndexer that needs to use something other
      # than a LongTensor here, and it's not clear how to signal that.  Maybe we'll need to
      # add a class method to TokenIndexer to tell us the type?  But we can worry about that
      # when there's a compelling use case for it.
      indexer_tensors = {key: torch.LongTensor(array) for key, array in padded_array.items()}
      tensors.update(indexer_tensors)
    return tensors

  @overrides
  def empty_field(self):
    text_field = TextField([], self.token_indexers)
    text_field._indexed_tokens = {}
    text_field._indexer_name_to_indexed_token = {}
    for indexer_name, indexer in self.token_indexers.items():
      array_keys = indexer.get_keys(indexer_name)
      for key in array_keys:
        text_field._indexed_tokens[key] = []
      text_field._indexer_name_to_indexed_token[indexer_name] = array_keys
    return text_field

  @overrides
  def batch_tensors(self, tensor_list: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    return batch_tensor_dicts(tensor_list)