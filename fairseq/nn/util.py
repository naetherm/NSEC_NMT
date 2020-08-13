# -*- coding: utf-8 -*-
'''
Copyright 2019, University of Freiburg.
Chair of Algorithms and Data Structures.
Markus Näther <naetherm@informatik.uni-freiburg.de>
'''

from collections import defaultdict
from typing import Any, Dict, List, Tuple

import numpy
import torch

from fairseq.common.checks import ConfigurationError

def has_tensor(obj) -> bool:
  """
  Given a possibly complex data structure,
  check if it has any torch.Tensors in it.
  """
  if isinstance(obj, torch.Tensor):
    return True
  elif isinstance(obj, dict):
    return any(has_tensor(value) for value in obj.values())
  elif isinstance(obj, (list, tuple)):
    return any(has_tensor(item) for item in obj)
  else:
    return False

def get_device_of(tensor: torch.Tensor) -> int:
  """
  Returns the device of the tensor.
  """
  if not tensor.is_cuda:
    return -1
  else:
    return tensor.get_device()

def batch_tensor_dicts(
  tensor_dicts: List[Dict[str, torch.Tensor]],
  remove_trailing_dimension: bool = False
) -> Dict[str, torch.Tensor]:
  key_to_tensors: Dict[str, List[torch.Tensor]] = defaultdict(list)
  for tensor_dict in tensor_dicts:
    for key, tensor in tensor_dict.items():
      key_to_tensors[key].append(tensor)
  batched_tensors = {}
  for key, tensor_list in key_to_tensors.items():
    batched_tensor = torch.stack(tensor_list)
    if remove_trailing_dimension and all(tensor.size(-1) == 1 for tensor in tensor_list):
      batched_tensor = batched_tensor.squeeze(-1)
    batched_tensors[key] = batched_tensor
  return batched_tensors

def get_lengths_from_binary_sequence_mask(mask: torch.Tensor):

  return mask.long().sum(-1)

def sort_batch_by_length(tensor: torch.Tensor, sequence_lengths: torch.Tensor):

  if not isinstance(tensor, torch.Tensor) or not isinstance(sequence_lengths, torch.Tensor):
    raise ConfigurationError("Both the tensor and sequence lengths must be torch.Tensors.")

  sorted_sequence_lengths, permutation_index = sequence_lengths.sort(0, descending=True)
  sorted_tensor = tensor.index_select(0, permutation_index)

  index_range = torch.arange(0, len(sequence_lengths), device=sequence_lengths.device)
  # This is the equivalent of zipping with index, sorting by the original
  # sequence lengths and returning the now sorted indices.
  _, reverse_mapping = permutation_index.sort(0, descending=False)
  restoration_indices = index_range.index_select(0, reverse_mapping)
  return sorted_tensor, sorted_sequence_lengths, restoration_indices, permutation_index

def move_to_device(obj, cuda_device: int):
  """
  Given a structure (possibly) containing Tensors on the CPU,
  move all the Tensors to the specified GPU (or do nothing, if they should be on the CPU).
  """
  # pylint: disable=too-many-return-statements
  if cuda_device < 0 or not has_tensor(obj):
    return obj
  elif isinstance(obj, torch.Tensor):
    return obj.cuda(cuda_device)
  elif isinstance(obj, dict):
    return {key: move_to_device(value, cuda_device) for key, value in obj.items()}
  elif isinstance(obj, list):
    return [move_to_device(item, cuda_device) for item in obj]
  elif isinstance(obj, tuple) and hasattr(obj, '_fields'):
    # This is the best way to detect a NamedTuple, it turns out.
    return obj.__class__(*[move_to_device(item, cuda_device) for item in obj])
  elif isinstance(obj, tuple):
    return tuple([move_to_device(item, cuda_device) for item in obj])
  else:
    return obj

def device_mapping(cuda_device: int):
  """
  In order to `torch.load()` a GPU-trained model onto a CPU (or specific GPU),
  you have to supply a `map_location` function. Call this with
  the desired `cuda_device` to get the function that `torch.load()` needs.
  """

  def inner_device_mapping(storage: torch.Storage, location) -> torch.Storage:  # pylint: disable=unused-argument
    if cuda_device >= 0:
      return storage.cuda(cuda_device)
    else:
      return storage

  return inner_device_mapping


def masked_softmax(
  vector: torch.Tensor,
  mask: torch.Tensor,
  dim: int = -1,
  memory_efficient: bool = False,
  mask_fill_value: float = -1e32) -> torch.Tensor:
  """
  ``torch.nn.functional.softmax(vector)`` does not work if some elements of ``vector`` should be
  masked.  This performs a softmax on just the non-masked portions of ``vector``.  Passing
  ``None`` in for the mask is also acceptable; you'll just get a regular softmax.
  ``vector`` can have an arbitrary number of dimensions; the only requirement is that ``mask`` is
  broadcastable to ``vector's`` shape.  If ``mask`` has fewer dimensions than ``vector``, we will
  unsqueeze on dimension 1 until they match.  If you need a different unsqueezing of your mask,
  do it yourself before passing the mask into this function.
  If ``memory_efficient`` is set to true, we will simply use a very large negative number for those
  masked positions so that the probabilities of those positions would be approximately 0.
  This is not accurate in math, but works for most cases and consumes less memory.
  In the case that the input vector is completely masked and ``memory_efficient`` is false, this function
  returns an array of ``0.0``. This behavior may cause ``NaN`` if this is used as the last layer of
  a model that uses categorical cross-entropy loss. Instead, if ``memory_efficient`` is true, this function
  will treat every element as equal, and do softmax over equal numbers.
  """
  if mask is None:
    result = torch.nn.functional.softmax(vector, dim=dim)
  else:
    mask = mask.float()
    while mask.dim() < vector.dim():
      mask = mask.unsqueeze(1)
    if not memory_efficient:
      # To limit numerical errors from large vector elements outside the mask, we zero these out.
      result = torch.nn.functional.softmax(vector * mask, dim=dim)
      result = result * mask
      result = result / (result.sum(dim=dim, keepdim=True) + 1e-13)
    else:
      masked_vector = vector.masked_fill((1 - mask).byte(), mask_fill_value)
      result = torch.nn.functional.softmax(masked_vector, dim=dim)
  return result



def get_text_field_mask(text_field_tensors: Dict[str, torch.Tensor],
                        num_wrapping_dims: int = 0) -> torch.LongTensor:
  """
  Takes the dictionary of tensors produced by a ``TextField`` and returns a mask
  with 0 where the tokens are padding, and 1 otherwise.  We also handle ``TextFields``
  wrapped by an arbitrary number of ``ListFields``, where the number of wrapping ``ListFields``
  is given by ``num_wrapping_dims``.
  If ``num_wrapping_dims == 0``, the returned mask has shape ``(batch_size, num_tokens)``.
  If ``num_wrapping_dims > 0`` then the returned mask has ``num_wrapping_dims`` extra
  dimensions, so the shape will be ``(batch_size, ..., num_tokens)``.
  There could be several entries in the tensor dictionary with different shapes (e.g., one for
  word ids, one for character ids).  In order to get a token mask, we use the tensor in
  the dictionary with the lowest number of dimensions.  After subtracting ``num_wrapping_dims``,
  if this tensor has two dimensions we assume it has shape ``(batch_size, ..., num_tokens)``,
  and use it for the mask.  If instead it has three dimensions, we assume it has shape
  ``(batch_size, ..., num_tokens, num_features)``, and sum over the last dimension to produce
  the mask.  Most frequently this will be a character id tensor, but it could also be a
  featurized representation of each token, etc.
  If the input ``text_field_tensors`` contains the "mask" key, this is returned instead of inferring the mask.
  TODO(joelgrus): can we change this?
  NOTE: Our functions for generating masks create torch.LongTensors, because using
  torch.ByteTensors  makes it easy to run into overflow errors
  when doing mask manipulation, such as summing to get the lengths of sequences - see below.
  >>> mask = torch.ones([260]).byte()
  >>> mask.sum() # equals 260.
  >>> var_mask = torch.autograd.V(mask)
  >>> var_mask.sum() # equals 4, due to 8 bit precision - the sum overflows.
  """
  if "mask" in text_field_tensors:
    return text_field_tensors["mask"]

  tensor_dims = [(tensor.dim(), tensor) for tensor in text_field_tensors.values()]
  tensor_dims.sort(key=lambda x: x[0])

  smallest_dim = tensor_dims[0][0] - num_wrapping_dims
  if smallest_dim == 2:
    token_tensor = tensor_dims[0][1]
    return (token_tensor != 0).long()
  elif smallest_dim == 3:
    character_tensor = tensor_dims[0][1]
    return ((character_tensor > 0).long().sum(dim=-1) > 0).long()
  else:
    raise ValueError("Expected a tensor with dimension 2 or 3, found {}".format(smallest_dim))

def get_dropout_mask(dropout_probability: float, tensor_for_masking: torch.Tensor):

  binary_mask = (torch.rand(tensor_for_masking.size()) > dropout_probability).to(tensor_for_masking.device)

  result = binary_mask.float().div(1.0 - dropout_probability)

  return result


def remove_sentence_boundaries(tensor: torch.Tensor,
                               mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
  """
  Remove begin/end of sentence embeddings from the batch of sentences.
  Given a batch of sentences with size ``(batch_size, timesteps, dim)``
  this returns a tensor of shape ``(batch_size, timesteps - 2, dim)`` after removing
  the beginning and end sentence markers.  The sentences are assumed to be padded on the right,
  with the beginning of each sentence assumed to occur at index 0 (i.e., ``mask[:, 0]`` is assumed
  to be 1).
  Returns both the new tensor and updated mask.
  This function is the inverse of ``add_sentence_boundary_token_ids``.
  Parameters
  ----------
  tensor : ``torch.Tensor``
      A tensor of shape ``(batch_size, timesteps, dim)``
  mask : ``torch.Tensor``
        A tensor of shape ``(batch_size, timesteps)``
  Returns
  -------
  tensor_without_boundary_tokens : ``torch.Tensor``
      The tensor after removing the boundary tokens of shape ``(batch_size, timesteps - 2, dim)``
  new_mask : ``torch.Tensor``
      The new mask for the tensor of shape ``(batch_size, timesteps - 2)``.
  """
  # TODO: matthewp, profile this transfer
  sequence_lengths = mask.sum(dim=1).detach().cpu().numpy()
  tensor_shape = list(tensor.data.shape)
  new_shape = list(tensor_shape)
  new_shape[1] = tensor_shape[1] - 2
  tensor_without_boundary_tokens = tensor.new_zeros(*new_shape)
  new_mask = tensor.new_zeros((new_shape[0], new_shape[1]), dtype=torch.long)
  for i, j in enumerate(sequence_lengths):
    if j > 2:
      tensor_without_boundary_tokens[i, :(j - 2), :] = tensor[i, 1:(j - 1), :]
      new_mask[i, :(j - 2)] = 1

  return tensor_without_boundary_tokens, new_mask

def add_sentence_boundary_token_ids(tensor: torch.Tensor,
                                    mask: torch.Tensor,
                                    sentence_begin_token: Any,
                                    sentence_end_token: Any) -> Tuple[torch.Tensor, torch.Tensor]:
  """
  Add begin/end of sentence tokens to the batch of sentences.
  Given a batch of sentences with size ``(batch_size, timesteps)`` or
  ``(batch_size, timesteps, dim)`` this returns a tensor of shape
  ``(batch_size, timesteps + 2)`` or ``(batch_size, timesteps + 2, dim)`` respectively.
  Returns both the new tensor and updated mask.
  Parameters
  ----------
  tensor : ``torch.Tensor``
      A tensor of shape ``(batch_size, timesteps)`` or ``(batch_size, timesteps, dim)``
  mask : ``torch.Tensor``
        A tensor of shape ``(batch_size, timesteps)``
  sentence_begin_token: Any (anything that can be broadcast in torch for assignment)
      For 2D input, a scalar with the <S> id. For 3D input, a tensor with length dim.
  sentence_end_token: Any (anything that can be broadcast in torch for assignment)
      For 2D input, a scalar with the </S> id. For 3D input, a tensor with length dim.
  Returns
  -------
  tensor_with_boundary_tokens : ``torch.Tensor``
      The tensor with the appended and prepended boundary tokens. If the input was 2D,
      it has shape (batch_size, timesteps + 2) and if the input was 3D, it has shape
      (batch_size, timesteps + 2, dim).
  new_mask : ``torch.Tensor``
      The new mask for the tensor, taking into account the appended tokens
      marking the beginning and end of the sentence.
  """
  # TODO: matthewp, profile this transfer
  sequence_lengths = mask.sum(dim=1).detach().cpu().numpy()
  tensor_shape = list(tensor.data.shape)
  new_shape = list(tensor_shape)
  new_shape[1] = tensor_shape[1] + 2
  tensor_with_boundary_tokens = tensor.new_zeros(*new_shape)
  if len(tensor_shape) == 2:
    tensor_with_boundary_tokens[:, 1:-1] = tensor
    tensor_with_boundary_tokens[:, 0] = sentence_begin_token
    for i, j in enumerate(sequence_lengths):
      tensor_with_boundary_tokens[i, j + 1] = sentence_end_token
    new_mask = (tensor_with_boundary_tokens != 0).long()
  elif len(tensor_shape) == 3:
    tensor_with_boundary_tokens[:, 1:-1, :] = tensor
    for i, j in enumerate(sequence_lengths):
      tensor_with_boundary_tokens[i, 0, :] = sentence_begin_token
      tensor_with_boundary_tokens[i, j + 1, :] = sentence_end_token
    new_mask = ((tensor_with_boundary_tokens > 0).long().sum(dim=-1) > 0).long()
  else:
    raise ValueError("add_sentence_boundary_token_ids only accepts 2D and 3D input")

  return tensor_with_boundary_tokens, new_mask