# -*- coding: utf-8 -*-
'''
Copyright 2019, University of Freiburg.
Chair of Algorithms and Data Structures.
Markus Näther <naetherm@informatik.uni-freiburg.de>
'''

from typing import List, Optional
from overrides import overrides
import logging

from fairseq.common.checks import ConfigurationError
from fairseq.training.base_trainer import BaseTrainer
from fairseq.training.multi_task_trainer import MultiTaskTrainer
from fairseq.common.params import Params
from fairseq.tasks.task import Task
from fairseq.models.fairseq_model import BaseFairseqModel

LOGGER = logging.getLogger(__name__)

@BaseTrainer.register("sampler_multi_task_trainer")
class SamplerMultiTaskTrainer(MultiTaskTrainer):

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
    min_lr: float = 0.00001,
    sampling_method: str = "proportional"
  ) -> None:
    """
    Constructor.
    """
    # First, call the super class constructor
    super(SamplerMultiTaskTrainer, self).__init__(
      model=model,
      task_list=task_list,
      optimizer_params=optimizer_params,
      lr_scheduler_params=lr_scheduler_params,
      patience=patience,
      num_epochs=num_epochs,
      serialization_dir=serialization_dir,
      cuda_device=cuda_device,
      gradn_norm=grad_norm,
      grad_clipping=grad_clipping,
      min_lr=min_lr
    )

    if sampling_method not in ["proportional", "uniform"]:
      raise ConfigurationError(f"the sampling method {sampling_method} is not known and not supported. We only support: `proportional` and `uniform`.")

    # Now everything else
    self.sampling_method = sampling_method

  @overrides
  def train(self, recover: bool = False):
    """
    """
    # TODO(naetherm): Implement this!
    pass


  @classmethod
  def from_params(cls, model: BaseFairseqModel, task_list: List[Task], serialization_dir: str, params: Params) -> 'SamplerMultiTaskTrainer':
    """
    """
    pass