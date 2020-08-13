# -*- coding: utf-8 -*-
'''
Copyright 2019, University of Freiburg.
Chair of Algorithms and Data Structures.
Markus Näther <naetherm@informatik.uni-freiburg.de>
'''

from typing import Any, Dict, List
import logging
import regex as re

import torch

from fairseq.common.params import Params
from fairseq.common.checks import ConfigurationError
from fairseq.common.registrable import Registrable

LOGGER = logging.getLogger(__name__)

class Optimizer(Registrable):

  @classmethod
  def from_params(cls, model_parameters: List, params: Params):

    if isinstance(params, str):
      optimizer = params
      params = Params({})
    else:
      optimizer = params.pop_choice("type", Optimizer.list_available())

    # make the parameter groups if need
    groups = params.pop("parameter_groups", None)
    if groups:
      # The input to the optimizer is list of dict.
      # Each dict contains a "parameter group" and groups specific options,
      # e.g., {'params': [list of parameters], 'lr': 1e-3, ...}
      # Any config option not specified in the additional options (e.g.
      # for the default group) is inherited from the top level config.
      # see: https://pytorch.org/docs/0.3.0/optim.html?#per-parameter-options
      #
      # groups contains something like:
      #"parameter_groups": [
      #       [["regex1", "regex2"], {"lr": 1e-3}],
      #       [["regex3"], {"lr": 1e-4}]
      #]
      #(note that the allennlp config files require double quotes ", and will
      # fail (sometimes silently) with single quotes ').

      # This is typed as as Any since the dict values other then
      # the params key are passed to the Optimizer constructor and
      # can be any type it accepts.
      # In addition to any parameters that match group specific regex,
      # we also need a group for the remaining "default" group.
      # Those will be included in the last entry of parameter_groups.
      parameter_groups: Any = [{'params': []} for _ in range(len(groups) + 1)]
      # add the group specific kwargs
      for k in range(len(groups)): # pylint: disable=consider-using-enumerate
        parameter_groups[k].update(groups[k][1].as_dict())

      regex_use_counts: Dict[str, int] = {}
      parameter_group_names: List[set] = [set() for _ in range(len(groups) + 1)]
      for name, param in model_parameters:
        # Determine the group for this parameter.
        group_index = None
        for k, group_regexes in enumerate(groups):
          for regex in group_regexes[0]:
            if regex not in regex_use_counts:
              regex_use_counts[regex] = 0
            if re.search(regex, name):
              if group_index is not None and group_index != k:
                raise ValueError("{} was specified in two separate parameter groups".format(name))
              group_index = k
              regex_use_counts[regex] += 1

        if group_index is not None:
            parameter_groups[group_index]['params'].append(param)
            parameter_group_names[group_index].add(name)
        else:
            # the default group
            parameter_groups[-1]['params'].append(param)
            parameter_group_names[-1].add(name)

      # log the parameter groups
      LOGGER.info("Done constructing parameter groups.")
      for k in range(len(groups) + 1):
        group_options = {key: val for key, val in parameter_groups[k].items()
                          if key != 'params'}
        LOGGER.info("Group %s: %s, %s", k,
                    list(parameter_group_names[k]),
                    group_options)
      # check for unused regex
      for regex, count in regex_use_counts.items():
        if count == 0:
          LOGGER.warning("When constructing parameter groups, "
                          " %s not match any parameter name", regex)

    else:
      parameter_groups = [param for name, param in model_parameters]

    # Log the number of parameters to optimize
    num_parameters = 0
    for parameter_group in parameter_groups:
      if isinstance(parameter_group, dict):
        num_parameters += sum(parameter.numel() for parameter in parameter_group["params"])
      else:
        num_parameters += parameter_group.numel()
    LOGGER.info("Number of trainable parameters: %s", num_parameters)

    # By default we cast things that e.g. look like floats to floats before handing them
    # to the Optimizer constructor, but if you want to disable that behavior you could add a
    #       "infer_type_and_cast": false
    # key to your "trainer.optimizer" config.
    infer_type_and_cast = params.pop_bool("infer_type_and_cast", True)
    params_as_dict = params.as_dict(infer_type_and_cast=infer_type_and_cast)
    subclass = Optimizer.by_name(optimizer)

    # If the optimizer subclass has a from_params, use it.
    if hasattr(subclass, 'from_params'):
      return subclass.from_params(parameter_groups, params=params)
    else:
      return subclass(parameter_groups, **params_as_dict) # type: ignore


from fairseq.optim.adam import FairseqAdam

Registrable._registry[Optimizer] = {
  "adam": torch.optim.Adam
}