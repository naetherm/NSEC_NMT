# -*- coding: utf-8 -*-
'''
Copyright 2019, University of Freiburg.
Chair of Algorithms and Data Structures.
Markus Näther <naetherm@informatik.uni-freiburg.de>
'''

from typing import Dict

from overrides import overrides
import logging
import os
import pathlib
import csv

import jsonpickle

from fairseq.common.util import START_SYMBOL, END_SYMBOL
from fairseq.data.instance import Instance
from fairseq.data.tokenizers.token import Token
from fairseq.data.tokenizers.tokenizer import Tokenizer
from fairseq.data.instance import Instance
from fairseq.data.tokenizers.word_tokenizer import WordTokenizer
from fairseq.data.token_indexers.token_indexer import TokenIndexer
from fairseq.data.token_indexers.single_id_token_indexer import SingleIdTokenIndexer
from fairseq.data.fields.text_field import TextField
from fairseq.data.dataset_readers.dataset_reader import DatasetReader

LOGGER = logging.getLogger(__name__)

@DatasetReader.register("seq2seq")
class Seq2SeqDatasetReader(DatasetReader):

  def __init__(
    self,
    source_tokenizer: Tokenizer = None,
    target_tokenizer: Tokenizer = None,
    source_token_indexers: Dict[str, TokenIndexer] = None,
    target_token_indexers: Dict[str, TokenIndexer] = None,
    source_add_start_token: bool = True,
    delimiter: str = "\t",
    lazy: bool = False
  ) -> None:
    super(Seq2SeqDatasetReader, self).__init__(lazy)

    self.source_tokenizer = source_tokenizer or WordTokenizer()
    self.target_tokenizer = target_tokenizer or self.source_tokenizer
    self.source_token_indexers = source_token_indexers or {"tokens": SingleIdTokenIndexer()}
    self.target_token_indexers = target_token_indexers or self.source_token_indexers
    self.source_add_start_token = source_add_start_token
    self.delimiter = delimiter

  @overrides
  def _read(self, file_path: str):
    with open(file_path, 'r') as fin:
      LOGGER.info("Reading instances from lines in file: %s", file_path)
      for line_num, row in enumerate(csv.reader(fin, delimiter=self.delimiter)):
        if len(row) != 2:
          pass # ERROR
        source_, target_ = row

        yield self.text_to_instance(source_, target_)

  @overrides
  def text_to_instance(self, source_string: str, target_string: str = None) -> Instance:
    tokenized_source = self.source_tokenizer.tokenize(source_string)
    if self.source_add_start_token:
      tokenized_source.insert(0, Token(START_SYMBOL))
    tokenized_source.append(Token(END_SYMBOL))

    source_field = TextField(tokenized_source, self.source_token_indexers)

    if target_string is not None:
      tokenized_target = self.target_tokenizer.tokenize(target_string)
      tokenized_target.insert(0, Token(START_SYMBOL))
      tokenized_target.append(Token(END_SYMBOL))
      target_field = TextField(tokenized_target, self.target_token_indexers)

      return Instance({"source_tokens": source_field, "target_tokens": target_field})
    else:
      return Instance({"source_tokens": source_field})