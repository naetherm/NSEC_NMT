# -*- coding: utf-8 -*-
'''
Copyright 2019, University of Freiburg.
Chair of Algorithms and Data Structures.
Markus Näther <naetherm@informatik.uni-freiburg.de>
'''

from typing import Deque, Iterable
from overrides import overrides
from collections import deque

import random 

import logging

from fairseq.common.util import lazy_groups_of
from fairseq.data.instance import Instance
from fairseq.data.iterators.data_iterator import DataIterator
from fairseq.data.dataset import Batch

logger = logging.getLogger(__name__)

@DataIterator.register("basic")
class BasicIterator(DataIterator):
  """
  Very basic data iterator that takes datasets, shuffles them and creates fixed sized batches 
  out of them.
  """

  @overrides
  def __create_batches(
    self,
    instances: Iterable[Instance],
    shuffle: bool
  ) -> Iterable[Batch]:
    """
    This method should return one epoch worth of batches.
    """
    for instance_list in self._memory_sized_lists(instances):
      if shuffle:
        random.shuffle(instance_list)
      
      iterator = iter(instance_list)
      excess: Deque[Instance] = deque()

      for batch_instances in lazy_groups_of(iterator, self._batch_size):
        for possibly_smaller_batch in self._ensure_batch_is_sufficiently_small(batch_instances, excess):
          batch = Batch(possibly_smaller_batch)
          yield batch
      if excess:
        yield Batch(excess)