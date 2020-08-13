# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import regex as re

SPACE_NORMALIZER = re.compile(r"\s+")


def tokenize_line(line, character_level: bool = True, word_level: bool = False):
    line = SPACE_NORMALIZER.sub(" ", line)
    line = line.strip()
    if character_level and not word_level:
      return list(line)
    if not character_level and word_level:
      return line.split()
    if character_level and word_level:
      temp = line.split()
      return [[t] + list(t) for t in temp]
      

def advanced_line_tokenizer(line, character_level: bool = True, word_level: bool = True):
    """
    This method ises the same tokenization regex as in NSEC and our benchmark. 

    > Input: "This is a sentence."

    > Output: [["This, "T", "h", "i", "s"], ["is", "i", "s"], ..., [".", "."]]
    """
    tokens = re.findall(r"(?:\d+,\d+)|(?:[\w'\u0080-\u9999]+(?:[-]+[\w'\u0080-\u9999]+)+)|(?:[\w\u0080-\u9999]+(?:[']+[\w\u0080-\u9999]+)+)|\b[_]|(?:[_]*[\w\u0080-\u9999]+(?=_\b))|(?:[\w\u0080-\u9999]+)|[^\w\s\p{Z}]", line, re.UNICODE)

    if character_level and not word_level:
      return [list(w) for w in tokens]
    if not character_level and word_level:
      return [[w] for w in tokens]
    if character_level and word_level:
      return [[w] + list(w) for w in tokens]