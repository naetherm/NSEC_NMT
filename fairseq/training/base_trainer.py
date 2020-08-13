# -*- coding: utf-8 -*-
'''
Copyright 2019, University of Freiburg.
Chair of Algorithms and Data Structures.
Markus Näther <naetherm@informatik.uni-freiburg.de>
'''

from typing import List
import logging

from fairseq.common.checks import ConfigurationError, check_for_gpu
from fairseq.common.registrable import Registrable
from fairseq.tasks.task import Task
from fairseq.models.fairseq_model import BaseFairseqModel

LOGGER = logging.getLogger(__name__)

class BaseTrainer(Registrable):
  """
  """

  default_implementation = "trainer"

  def __init__(
    self,
    task_list,
    serialization_dir: str,
    cuda_device: int = -1
  ) -> None:
    check_for_gpu(cuda_device)
    
    self.task_list = task_list
    self.serialization_dir = serialization_dir
    self.cuda_device = cuda_device

  def train(self, recover: bool = False):
    """
    Train a model.
    """
    raise NotImplementedError