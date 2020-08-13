# -*- coding: utf-8 -*-
'''
Copyright 2019, University of Freiburg.
Chair of Algorithms and Data Structures.
Markus Näther <naetherm@informatik.uni-freiburg.de>
'''

from fairseq.data.fields.field import Field, DataArray

class SequenceField(Field[DataArray]):

  def sequence_length(self) -> int:
    raise NotImplementedError

  def empty_field(self) -> 'SequenceField':
    raise NotImplementedError