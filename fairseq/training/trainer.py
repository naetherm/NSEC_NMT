# -*- coding: utf-8 -*-
'''
Copyright 2019, University of Freiburg.
Chair of Algorithms and Data Structures.
Markus Näther <naetherm@informatik.uni-freiburg.de>
'''

from typing import List
import time
from overrides import overrides
import logging

from fairseq.common.params import Params
from fairseq.training.base_trainer import BaseTrainer
from fairseq.optim.optimizer import Optimizer
from fairseq.tasks.task import Task
from fairseq.models.fairseq_model import BaseFairseqModel

@BaseTrainer.register("trainer")
class Trainer(BaseTrainer):

  def __init__(
    self,
    model: BaseFairseqModel,
    task_list: List[Task],
    serialization_dir: str,
    cuda_device: int = -1,
    grad_clipping: float = 0.1,
    grad_norm: float = 5.0,
    min_lr: float = 1e-7,
    num_epochs: int = 100,
    patience: int = 5,
    optimizer: Optimizer = None
  ) -> None:
    # Call the super class
    super(Trainer, self).__init__(task_list, serialization_dir, cuda_device=cuda_device)

    self.model = model

  def train(self, recover: bool = False):
    """
    Train the model.
    """
    # Record the start time
    start_training_time = time.time()

    if recover:
      pass
    else:
      n_epoch, should_stop = 0, False

  @classmethod
  def from_params(cls, model: BaseFairseqModel, task_list: List[Task], serialization_dir: str, params: Params) -> 'Trainer':
    """
    """
    cuda_device = params.pop_int("cuda_device", -1)
    grad_clipping = params.pop_float("grad_clipping", 0.1)
    grad_norm = params.pop_float("grad_norm", 5.0)
    min_lr = params.pop_float("min_lr", 1e-7)
    num_epochs = params.pop_int("num_epochs", 100)
    patience = params.pop_int("patience", 5)

    optimizer_params = params.pop("optimizer", None)
    parameters_to_train = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    optimizer_ = Optimizer.from_params(model_parameters=parameters_to_train, params=optimizer_params)

    return cls(
      model=model,
      task_list=task_list,
      serialization_dir=serialization_dir,
      cuda_device=cuda_device,
      grad_clipping=grad_clipping,
      grad_norm=grad_norm,
      min_lr=min_lr,
      num_epochs=num_epochs,
      patience=patience,
      optimizer=optimizer_
    )