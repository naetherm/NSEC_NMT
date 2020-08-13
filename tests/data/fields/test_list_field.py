# -*- coding: utf-8 -*-
'''
Copyright 2019, University of Freiburg.
Chair of Algorithms and Data Structures.
Markus Näther <naetherm@informatik.uni-freiburg.de>
'''

from collections import defaultdict
from typing import Dict, List

import argparse
import unittest

import numpy
import torch

import tests.utils as test_utils

from fairseq.data.fields import IndexField, LabelField, ListField, SequenceLabelField, TextField
from fairseq.data.dictionary import Dictionary


#class TestListField(unittest.TestCase):