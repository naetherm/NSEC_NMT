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

from fairseq.data.tokenizers.word_filter import *

class TestWordFilter(unittest.TestCase):

  def setUp(self):

    super(TestWordFilter, self).setUp()

    self.sentence = ["this", "sentence", "(", "a", "small", "one", ")", "has", ",", "some", "punctuations", "."]