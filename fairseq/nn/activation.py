# -*- coding: utf-8 -*-
'''
Copyright 2019, University of Freiburg.
Chair of Algorithms and Data Structures.
Markus Näther <naetherm@informatik.uni-freiburg.de>
'''

import torch

from fairseq.common.registrable import Registrable

class Activation(Registrable):
  """
  The class ``Activation`` is just a slight wrapper around the default 
  PyTorch activation methods, so the activation methods are callable during 
  the initialization of a configuration description file.

  Never the less it is also possible to inherit from this class and implement
  your own activation function.
  """

  def __call__(self, tensor: torch.Tensor) -> torch.Tensor:

    raise NotImplementedError

Registrable._registry[Activation] = {  # type: ignore
  "linear": lambda: lambda x: x,
  "relu": torch.nn.ReLU,
  "relu6": torch.nn.ReLU6,
  "elu": torch.nn.ELU,
  "prelu": torch.nn.PReLU,
  "leaky_relu": torch.nn.LeakyReLU,
  "threshold": torch.nn.Threshold,
  "hardtanh": torch.nn.Hardtanh,
  "sigmoid": torch.nn.Sigmoid,
  "tanh": torch.nn.Tanh,
  "log_sigmoid": torch.nn.LogSigmoid,
  "softplus": torch.nn.Softplus,
  "softshrink": torch.nn.Softshrink,
  "softsign": torch.nn.Softsign,
  "tanhshrink": torch.nn.Tanhshrink,
}