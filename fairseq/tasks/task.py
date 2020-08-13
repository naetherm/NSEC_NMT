# -*- coding: utf-8 -*-
'''
Copyright 2019, University of Freiburg.
Chair of Algorithms and Data Structures.
Markus Näther <naetherm@informatik.uni-freiburg.de>
'''

from typing import List
import logging

from fairseq.common.checks import ConfigurationError
from fairseq.common.params import Params
#from fairseq.training.util import datasets_from_params

LOGGER = logging.getLogger(__name__)


from typing import Dict, Iterable
from fairseq.data.dataset_readers.dataset_reader import DatasetReader
from fairseq.data.instance import Instance

def datasets_from_params(
  params: Params
) -> Dict[str, Iterable[Instance]]:
  # Receive the configuration for the dataset reader to use
  dataset_reader_params = params.pop("dataset_reader")

  # Initialize the dataset reader
  dataset_reader = DatasetReader.from_params(dataset_reader_params)

  # We will definitively need a training data path
  training_data_path = params.pop("train_data_path")
  LOGGER.info(f"reading training data from path '{training_data_path}'")
  train_data = dataset_reader.read(training_data_path)

  datasets: Dict[str, Iterable[Instance]] = {"train": train_data}

  # Now the optional stuff: validation and test datasets
  validation_data_path = params.pop("validation_data_path", None)
  if validation_data_path is not None:
    LOGGER.info(f"reading validation data from path '{validation_data_path}'")
    validation_data = dataset_reader.read(validation_data_path)
    datasets["validation"] = validation_data
  
  test_data_path = params.pop("test_data_path", None)
  if test_data_path is not None:
    LOGGER.info(f"reading test data from path '{test_data_path}'")
    test_data = dataset_reader.read(test_data_path)
    datasets["test"] = test_data

  # Done, now return the dictionary of all datasets
  return datasets

class Task(object):
  """
  The class ``Task`` describes all the necessary information that is required
  for a unique task.

  We require our own implementation because we remove all task specific mambo jambo
  as it is with the current FairseqTask implementation. ``Task`` will now be just 
  a small wrapper around the data required for creating and feeding a model.
  """

  def __init__(
    self,
    name: str,
    validation_metric_name: str,
    validation_metric_decreases: bool = False,
    evaluate_on_test: bool = False
  ) -> None:
    """
    Constructor.
    """
    self.name = name

    self.train_data = None
    self.validation_data = None
    self.test_data = None
    self.train_instances = None
    self.validation_instances = None
    self.test_instances = None

    self.data_iterator = None

    self.evaluate_on_test = False

  def set_data_iterator(self, data_iterator):
    """
    """
    if data_iterator is not None:
      self.data_iterator = data_iterator
    else:
      raise ConfigurationError(f"data_iterator cannot be None - Task '{self.name}'")

  def get_data_iterator(self):
    """
    """
    return self.data_iterator

  def setup_data(self, params: Params):
    """
    This method is responsible for fetching the dataset information from the given 
    parameters and setup everything related to the data.
    """
    all_datasets = datasets_from_params(params)
    datasets_for_vocab_creation = set(params.pop("datasets_for_vocab_creation", all_datasets))

    LOGGER.info(f"datasets_for_vocab_creation: {datasets_for_vocab_creation}")

    for dataset in datasets_for_vocab_creation:
      if dataset not in all_datasets:
        raise ConfigurationError(f"the dataset {dataset} is not known in 'all_datasets")
    
    # TODO(naetherm): Implement me!
    instances_for_vocab_creation = ()

    self.instances_for_vocab_creation = instances_for_vocab_creation
    self.datasets_for_vocab_creation = datasets_for_vocab_creation

    if "train" in all_datasets.keys():
      self.train_data = all_datasets["train"]
      self.train_instances = sum(1 for e in self.train_data)
    if "validation" in all_datasets.keys():
      self.validation_data = all_datasets["validation"]
      self.validation_instances = sum(1 for e in self.validation_data)
    if "test" in all_datasets.keys():
      self.test_data = all_datasets["test"]
      self.test_instances = sum(1 for e in self.test_data)

    # Security check: If we want to evaluate on the test data we _must_ have test data!
    if self.evaluate_on_test:
      assert self.test_data is not None
    
    return self.instances_for_vocab_creation, self.datasets_for_vocab_creation

  @classmethod
  def from_params(cls, params: Params) -> 'Task':
    """
    Create a task instance from parameters.
    """
    task_name = params.pop("task_name", "ensec")
    validation_metric_name = params.pop("validation_metric_name", None)
    validation_metric_decreases = params.pop_bool("validation_metric_decreases", False)
    evaluate_on_test = params.pop_bool("evaluate_on_test", False)
    
    params.assert_empty(cls.__name__)
    
    return cls(
      name=task_name,
      validation_metric_name=validation_metric_name,
      validation_metric_decreases=validation_metric_decreases,
      evaluate_on_test=evaluate_on_test
    )