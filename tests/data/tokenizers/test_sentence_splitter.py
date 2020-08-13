# -*- coding: utf-8 -*-
'''
Copyright 2019, University of Freiburg.
Chair of Algorithms and Data Structures.
Markus Näther <naetherm@informatik.uni-freiburg.de>
'''

import argparse
import unittest

import torch

import tests.utils as test_utils

from fairseq.data.tokenizers.sentence_splitter import SentenceSplitter, SpacySentenceSplitter

class TestSentenceSplitter(unittest.TestCase):

  def setUp(self):

    super(TestSentenceSplitter, self).setUp()

    
