# -*- coding: utf-8 -*-
'''
Copyright 2019, University of Freiburg.
Chair of Algorithms and Data Structures.
Markus Näther <naetherm@informatik.uni-freiburg.de>
'''

from typing import List
from overrides import overrides

import spacy

from fairseq.common.registrable import Registrable

from fairseq.common.util import get_spacy_model


class SentenceSplitter(Registrable):

  default_implementation = "spacy"

  def split_sentences(self, text: str) -> List[str]:
    raise NotImplementedError

  def batch_split_sentences(self, texts: List[str]) -> List[List[str]]:
    return [self.split_sentences(s) for s in texts]


@SentenceSplitter.register("spacy")
class SpacySentenceSplitter(SentenceSplitter):

  def __init__(
    self,
    language: str = 'en_core_web_sm',
    rule_based: bool = False) -> None:
    self.spacy = get_spacy_model(language, parse=not rule_based, ner=False, pos_tags=False)

    if rule_based:
      sbd_name = 'sbd' if spacy.__version__ < '2.1' else 'sentencizer'
      if not self.spacy.has_pipe(sbd_name):
        sbd = self.spacy.create_pipe(sbd_name)
        self.spacy.add_pipe(sbd)
  
  @overrides
  def split_sentences(self, text: str) -> List[str]:
    return [s.string.strip() for s in self.spacy(text).sents]

  @overrides
  def batch_split_sentences(self, texts: List[str]) -> List[List[str]]:
    return [[s.string.strip() for s in doc.sents] for doc in self.spacy.pipe(texts)]