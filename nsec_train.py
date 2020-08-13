# -*- coding: utf-8 -*-
'''
Copyright 2019, University of Freiburg.
Chair of Algorithms and Data Structures.
Markus Näther <naetherm@informatik.uni-freiburg.de>
'''

from typing import Any, List, Dict, Tuple
import itertools
import os
import json
import regex as re
from copy import deepcopy
import logging

import argparse
import numpy as np

import torch

from fairseq import utils
from fairseq.common.checks import ConfigurationError
from fairseq.common.params import Params
from fairseq.data.dictionary import Dictionary
from fairseq.models.fairseq_model import BaseFairseqModel
from fairseq.training.base_trainer import BaseTrainer
from fairseq.tasks.task import Task

# DEBUG
from fairseq.data.dataset_readers.dataset_reader import DatasetReader
from fairseq.data.dataset_readers.spell_correction_reader import SpellCorrectionDatasetReader

try:
  from polyaxon_client.tracking import Experiment
  from polyaxon_client.tracking import get_outputs_path
  import polyaxon_client.settings as pst
  IN_CLUSTER = pst.IN_CLUSTER
except:
  IN_CLUSTER = False

LOGGER = logging.getLogger(__name__)

def create_serialization_dir(serialization_dir: str):
  if os.path.exists(serialization_dir):
    return
  else:
    os.makedirs(serialization_dir, exist_ok=True)

def train(trainer: BaseTrainer, recover: bool = False):

  LOGGER.info("start training ...")
  trainer.train(recover=recover)
  
  serialization_dir = trainer.serialization_dir
  task_list = trainer.task_list
  model = trainer.model

  for task in task_list:
    if not task.evaluate_on_test:
      continue

    LOGGER.info(f"the task '{task.name}' will be evaluated by using the best epoch weights.")

    assert (
      task.test_data is not None
    ), f"the task '{task.name}' wants to be evaluated on the test data but there is no such data available"

    LOGGER.info(f"loading the best epoch weights for the task '{task.name}")
    best_model_state_path = os.path.join(serialization_dir, "best_{}.th".format(task.name))
    best_model_state = torch.load(best_model_state_path)
    best_model = model
    best_model.load_state_dict(state_dict=best_model_state)

  LOGGER.info("finished training.")

def tasks_and_vocab_from_params(params: Params, serialization_dir: str) -> Tuple[List[Task], Dictionary]:
  """
  """
  task_list = []
  instances_for_vocab_creation = itertools.chain()
  datasets_for_vocab_creation = {}
  task_keys = [key for key in params.keys() if re.search("^task_", key)]

  for key in task_keys:
    LOGGER.info("Creating task '{}'".format(key))
    task_params = params.pop(key)
    task_description = task_params.pop("task_description")
    task_data_params = task_params.pop("data_params")

    task = Task.from_params(params=task_description)
    task_list.append(task)

    task_instances_for_vocab, task_datasets_for_vocab = task.setup_data(params=task_data_params)
    instances_for_vocab_creation = itertools.chain(instances_for_vocab_creation, task_instances_for_vocab)
    datasets_for_vocab_creation[task.name] = task_datasets_for_vocab

  # Create and save the dictionary
  for task_name, task_dataset_list in datasets_for_vocab_creation.items():
    LOGGER.info("creating dictionary for '{} from '{}'".format(task_name, ', '.join(task_dataset_list)))

  LOGGER.info('fitting dictionary from dataset')
  vocab = Dictionary.from_params(params.pop("dictionary", {}), instances_for_vocab_creation)

  # vocab save_to_files

  return task_list, vocab

def main():
  """
  """
  parser = argparse.ArgumentParser()
  parser.add_argument('--serialization_dir', type=str, 
                      help='The directory where to save trained models, etc.')
  parser.add_argument('--params', type=str, 
                      help='path to the parameter file describing the tasks to train.')
  parser.add_argument('--seed', type=int, default=1,
                      help='The random seed to use for the initialization of PyTorch and numpy.')
  parser.add_argument('--recover', action='store_true',
                      help='Recover from a previous experiment?')
  args = parser.parse_args()

  # Import user defined modules
  utils.import_user_module(args)

  # If we are in polyaxon redirect 
  if IN_CLUSTER:
    args.serialization_dir = get_outputs_path()

  # Set the random seed
  np.random.seed(args.seed)
  torch.manual_seed(args.seed)

  # Read the parameter file
  params = Params.from_file(args.params)
  serialization_dir = args.serialization_dir

  # Create the serialization directory
  create_serialization_dir(serialization_dir)
  
  # Write the parameter file to the output directory
  with open(os.path.join(serialization_dir, 'config.json'), 'w') as fout:
    json.dump(deepcopy(params).as_dict(quiet=True), fout, indent=2)


  # Call the tasks_and_vocab_from_params method
  tasks, vocab = tasks_and_vocab_from_params(params=params, serialization_dir=serialization_dir)

  # Load the data iterator for all tasks

  # Create the model
  model_params = params.pop("model")
  model = BaseFairseqModel.from_params(vocab=vocab, params=model_params)

  LOGGER.info("created model")
  print("created model: {}".format(model))

  # Finally, create an instance of the required trainer
  trainer_params = params.pop("trainer")
  # TODO(naetherm): Dependent on the trainer type ...
  trainer = BaseTrainer.from_params(model=model, task_list=tasks, serialization_dir=serialization_dir, params=trainer_params)

  # Everything is set up, start the training
  train(trainer)

if __name__ == '__main__':
  main()