# -*- coding: utf-8 -*-
'''
Copyright 2019, University of Freiburg.
Chair of Algorithms and Data Structures.
Markus Näther <naetherm@informatik.uni-freiburg.de>
'''

from typing import List, Optional
from overrides import overrides
import logging

from fairseq.tasks import Task
from fairseq.training.base_trainer import BaseTrainer
from fairseq.common.params import Params
from fairseq.models.fairseq_model import BaseFairseqModel

@BaseTrainer.register("multi_task_trainer")
class MultiTaskTrainer(BaseTrainer):

  def __init__(
    self,
    model: BaseFairseqModel,
    task_list: List[Task],
    optimizer_params: Params,
    lr_scheduler_params: Params,
    patience: Optional[int] = None,
    num_epochs: int = 50,
    serialization_dir: str = None,
    cuda_device: int = -1,
    grad_norm: Optional[float] = None,
    grad_clipping: Optional[float] = None,
    min_lr: float = 0.00001
  ) -> None:
    """
    Constructor.
    """
    # Call the super class
    super(MultiTaskTrainer, self).__init__(
      task_list=task_list, 
      serialization_dir=serialization_dir, 
      cuda_device=cuda_device
    )

    # Now, everything else
    self.model = model
    self.num_tasks = len(self.task_list)
    self.optimizer_params = optimizer_params

  @overrides
  def train(self):
    raise NotImplementedError


  @classmethod
  def from_params(
    cls, 
    model: BaseFairseqModel,
    task_list: List[Task],
    serialization_dir: str,
    params: Params
  ) -> 'MultiTaskTrainer':
    """
    Static class method that constructs a multi task trainer, based on the 
    description given in ``params``.
    """
    choices = params.pop_choice("type", cls.list_available())

    return cls.by_name(choices).from_params(
      model=model,
      task_list=task_list,
      serialization_dir=serialization_dir,
      params=params
    )