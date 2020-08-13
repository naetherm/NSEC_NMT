# -*- coding: utf-8 -*-
'''
Copyright 2019, University of Freiburg.
Chair of Algorithms and Data Structures.
Markus Näther <naetherm@informatik.uni-freiburg.de>
'''

import os

from typing import Dict, Tuple, List, Callable, Any, Iterable, Iterator, TypeVar
from itertools import islice

import spacy
from spacy.cli.download import download as spacy_download
from spacy.language import Language as SpacyModelType

import random
import numpy
import torch

import logging

from fairseq.common.params import Params

logger = logging.getLogger(__name__)

LOADED_SPACY_MODELS: Dict[Tuple[str, bool, bool, bool], SpacyModelType] = {}

A = TypeVar('A')

START_SYMBOL = '@start@'
END_SYMBOL = '@end@'

def get_spacy_model(spacy_model_name: str, pos_tags: bool, parse: bool, ner: bool) -> SpacyModelType:
  """
  In order to avoid loading spacy models a whole bunch of times, we'll save references to them,
  keyed by the options we used to create the spacy model, so any particular configuration only
  gets loaded once.
  """

  options = (spacy_model_name, pos_tags, parse, ner)
  if options not in LOADED_SPACY_MODELS:
    disable = ['vectors', 'textcat']
    if not pos_tags:
      disable.append('tagger')
    if not parse:
      disable.append('parser')
    if not ner:
      disable.append('ner')
    try:
      spacy_model = spacy.load(spacy_model_name, disable=disable)
    except OSError:
      logger.warning(f"Spacy models '{spacy_model_name}' not found.  Downloading and installing.")
      spacy_download(spacy_model_name)
      # NOTE(mattg): The following four lines are a workaround suggested by Ines for spacy
      # 2.1.0, which removed the linking that was done in spacy 2.0.  importlib doesn't find
      # packages that were installed in the same python session, so the way `spacy_download`
      # works in 2.1.0 is broken for this use case.  These four lines can probably be removed
      # at some point in the future, once spacy has figured out a better way to handle this.
      # See https://github.com/explosion/spaCy/issues/3435.
      from spacy.cli import link
      from spacy.util import get_package_path
      package_path = get_package_path(spacy_model_name)
      link(spacy_model_name, spacy_model_name, model_path=package_path)
      spacy_model = spacy.load(spacy_model_name, disable=disable)

    LOADED_SPACY_MODELS[options] = spacy_model
  return LOADED_SPACY_MODELS[options]


def get_file_extension(path: str, dot=True, lower: bool = True):
  ext = os.path.splitext(path)[1]
  ext = ext if dot else ext[1:]
  return ext.lower() if lower else ext


def pad_sequence_to_length(
  seq: List, 
  desired_len: int, 
  default_value: Callable[[], Any] = lambda: 0, 
  padding_on_right: bool = True) -> List:
  """
  This helper method takes a list instance of pads it to the desired_len.
  Keep in mind that the original list is untouched.
  """

  if padding_on_right:
    padded_seq = seq[:desired_len]
  else:
    padded_seq = seq[-desired_len:]
  
  # Now pad with default values until reaching the desired length
  for _ in range(desired_len - len(padded_seq)):
    if padding_on_right:
      padded_seq.append(default_value())
    else:
      padded_seq.insert(0, default_value())
  # Done
  return padded_seq

def namespace_match(pattern: str, namespace: str):
  if pattern[0] == '*' and namespace.endswith(pattern[1:]):
    return True
  elif pattern == namespace:
    return True
  return False


def ensure_list(iterable: Iterable[A]) -> List[A]:
  if isinstance(iterable, list):
    return iterable
  else:
    return list(iterable)

def flatten_filename(file_path: str) -> str:
  return file_path.replace('/', '_SLASH_')

def is_lazy(iterable: Iterable[A]) -> bool:
  return not isinstance(iterable, list)

def lazy_groups_of(iterator: Iterator[A], group_size: int) -> Iterator[List[A]]:
  """
  Takes an iterator and batches the individual instances into lists of the
  specified size. The last list may be smaller if there are instances left over.
  """
  return iter(lambda: list(islice(iterator, 0, group_size)), [])

def prepare_environment(params: Params):
  """
  Sets random seeds for reproducible experiments. This may not work as expected
  if you use this from within a python project in which you have already imported Pytorch.
  If you use the scripts/run_model.py entry point to training models with this library,
  your experiments should be reasonably reproducible. If you are using this from your own
  project, you will want to call this function before importing Pytorch. Complete determinism
  is very difficult to achieve with libraries doing optimized linear algebra due to massively
  parallel execution, which is exacerbated by using GPUs.
  Parameters
  ----------
  params: Params object or dict, required.
      A ``Params`` object or dict holding the json parameters.
  """
  seed = params.pop_int("random_seed", 13370)
  numpy_seed = params.pop_int("numpy_seed", 1337)
  torch_seed = params.pop_int("pytorch_seed", 133)

  if seed is not None:
    random.seed(seed)
  if numpy_seed is not None:
    numpy.random.seed(numpy_seed)
  if torch_seed is not None:
    torch.manual_seed(torch_seed)
    # Seed all GPUs with the same seed if available.
    if torch.cuda.is_available():
      torch.cuda.manual_seed_all(torch_seed)

def sanitize(x: Any) -> Any:  # pylint: disable=invalid-name,too-many-return-statements
  """
  Sanitize turns PyTorch and Numpy types into basic Python types so they
  can be serialized into JSON.
  """
  from fairseq.data.tokenizers.token import Token
  if isinstance(x, (str, float, int, bool)):
    # x is already serializable
    return x
  elif isinstance(x, torch.Tensor):
    # tensor needs to be converted to a list (and moved to cpu if necessary)
    return x.cpu().tolist()
  elif isinstance(x, numpy.ndarray):
    # array needs to be converted to a list
    return x.tolist()
  elif isinstance(x, numpy.number):  # pylint: disable=no-member
    # NumPy numbers need to be converted to Python numbers
    return x.item()
  elif isinstance(x, dict):
    # Dicts need their values sanitized
    return {key: sanitize(value) for key, value in x.items()}
  elif isinstance(x, (spacy.tokens.Token, Token)):
    # Tokens get sanitized to just their text.
    return x.text
  elif isinstance(x, (list, tuple)):
    # Lists and Tuples need their values sanitized
    return [sanitize(x_i) for x_i in x]
  elif x is None:
    return "None"
  elif hasattr(x, 'to_json'):
    return x.to_json()
  else:
    raise ValueError(f"Cannot sanitize {x} of type {type(x)}. "
                      "If this is your own custom class, add a `to_json(self)` method "
                      "that returns a JSON-like object.")