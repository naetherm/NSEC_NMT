# -*- coding: utf-8 -*-
'''
Copyright 2019, University of Freiburg.
Chair of Algorithms and Data Structures.
Markus Näther <naetherm@informatik.uni-freiburg.de>
'''

from typing import Iterable, Iterator, Callable

import logging
import os
import pathlib

import jsonpickle

from fairseq.common.registrable import Registrable

from fairseq.data.instance import Instance
from fairseq.common.tqdm import Tqdm
from fairseq.common.util import flatten_filename

logger = logging.getLogger(__name__)

class LazyInstances(Iterable):

  def __init__(
    self,
    instance_generator: Callable[[], Iterable[Instance]],
    cache_file: str = None,
    deserialize: Callable[[str], Instance] = None,
    serialize: Callable[[Instance], str] = None
  ) -> None:
    super().__init__()
    self.instance_generator = instance_generator
    self.cache_file = cache_file
    self.deserialize = deserialize
    self.serialize = serialize

  def __iter__(self) -> Iterator[Instance]:
    if self.cache_file is not None and os.path.exists(self.cache_file):
      with open(self.cache_file) as fin:
        for line in fin:
          yield self.deserialize(line)
    
    elif self.cache_file is not None:
      with open(self.cache_file, 'w') as fout:
        for instance in self.instance_generator():
          fout.write(self.serialize(instance))
          fout.write("\n")
          yield instance
    
    else:
      instances = self.instance_generator()
      
      yield from instances


class DatasetReader(Registrable):

  def __init__(self, lazy: bool = False) -> None:
    self.lazy = lazy
    self.cache_dir: pathlib.Path = None

  def cache_data(self, cache_dir: str) -> None:
    self.cache_dir = pathlib.Path(cache_dir)
    os.makedirs(self.cache_dir, exist_ok=True)

  def read(self, file_path: str) -> Iterable[Instance]:
    is_lazy = getattr(self, 'lazy', None)

    if is_lazy is None:
      logger.warning("Attribute lazy is unknown, did you call the super constructor?")
    
    if self.cache_dir:
      pass
    else:
      cache_file = None

    if is_lazy:
      return LazyInstances(lambda: self._read(file_path), cache_file, self.deserialize_instance, self.serialize_instance)
    else:
      if cache_file and os.path.exists(cache_file):
        instances = self._instances_from_cache_file(cache_file)
      else:
        instances = self._read(file_path)
      
      if not isinstance(instances, list):
        instances = [i for i in Tqdm.tqdm(instances)]
      
      # And finally we write to the cache if we need to.
      if cache_file and not os.path.exists(cache_file):
        logger.info(f"Caching instances to {cache_file}")
        with open(cache_file, 'w') as cache:
          for instance in Tqdm.tqdm(instances):
            cache.write(self.serialize_instance(instance) + '\n')
      return instances

  def _read(self, file_path: str) -> Iterable[Instance]:
    raise NotImplementedError

  def _instances_from_cache_file(self, cache_file: str) -> Iterable[Instance]:
    with open(cache_file, 'r') as fin:
      for line in fin:
        yield self.deserialize_instance(line.strip())

  def _get_cache_location_for_file_path(self, file_path: str) -> str:
    return str(self.cache_dir / flatten_filename(str(file_path)))

  def text_to_instance(self, *inputs) -> Instance:
    raise NotImplementedError

  def serialize_instance(self, instance: Instance) -> str:
    return jsonpickle.dumps(instance)

  def deserialize_instance(self, string: str) -> Instance:
    return jsonpickle.loads(string)