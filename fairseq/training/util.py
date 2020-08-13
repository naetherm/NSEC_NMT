# -*- coding: utf-8 -*-
'''
Copyright 2019, University of Freiburg.
Chair of Algorithms and Data Structures.
Markus Näther <naetherm@informatik.uni-freiburg.de>
'''

from typing import Dict, Iterable
import logging

from fairseq.common.params import Params
from fairseq.data.dataset_readers.dataset_reader import DatasetReader
from fairseq.data.instance import Instance

LOGGER = logging.getLogger(__name__)


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