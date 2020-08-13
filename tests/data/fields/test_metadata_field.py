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

from fairseq.data.fields import MetadataField
from fairseq.data.dictionary import Dictionary


class TestMetaDataField(unittest.TestCase):
  def test_mapping_words_with_dict(self):
    field = MetadataField({"a": 1, "b": [2]})

    assert "a" in field
    assert field["a"] == 1
    assert len(field) == 2 

    keys = {k for k in field}
    assert keys == {"a", "b"}

    values = [v for v in field.values()]
    assert len(values) == 2
    assert 1 in values
    assert [2] in values