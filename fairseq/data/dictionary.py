# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import Counter, defaultdict
from multiprocessing import Pool
import os
import codecs
import copy
import logging

from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Union
from typing import TextIO # pylint: disable=unused-import

import torch

from fairseq.common.checks import ConfigurationError
from fairseq.common.registrable import Registrable
from fairseq.common.params import Params
from fairseq.common.tqdm import Tqdm
from fairseq.data import instance as adi
from fairseq.tokenizer import tokenize_line
from fairseq.binarizer import safe_readline
from fairseq.data import data_utils

from fairseq.common.util import namespace_match

LOGGER = logging.getLogger(__name__)

DEFAULT_NON_PADDED_NAMESPACES = ("*tags", "*labels")
NAMESPACE_PADDING_FILE = 'non_padded_namespaces.txt'


def _read_pretrained_tokens(embeddings_file_uri: str) -> List[str]:
    """
    Helper method for loading pretrained tokens from a given file. This is usually
    the case for e.g. GloVe files, containing the token and the embedding of that token.

    Parameters
    ----------
    embeddings_file_url : ``str`` The path to the embedding file to load.
    """
    # Moving this import to the top breaks everything (cycling import, I guess)
    from allennlp.modules.token_embedders.embedding import EmbeddingsTextFile

    LOGGER.info('Reading pretrained tokens from: %s', embeddings_file_uri)
    tokens: List[str] = []
    with EmbeddingsTextFile(embeddings_file_uri) as embeddings_file:
        for line_number, line in enumerate(Tqdm.tqdm(embeddings_file), start=1):
            token_end = line.find(' ')
            if token_end >= 0:
                token = line[:token_end]
                tokens.append(token)
            else:
                line_begin = line[:20] + '...' if len(line) > 20 else line
                LOGGER.warning(f'Skipping line number %d: %s', line_number, line_begin)
    return tokens

def pop_max_vocab_size(params: Params) -> Union[int, Dict[str, int]]:
    """
    max_vocab_size limits the size of the vocabulary, not including the @@UNKNOWN@@ token.
    max_vocab_size is allowed to be either an int or a Dict[str, int] (or nothing).
    But it could also be a string representing an int (in the case of environment variable
    substitution). So we need some complex logic to handle it.
    """
    size = params.pop("max_vocab_size", None, keep_as_dict=True)

    if isinstance(size, dict):
        # This is the Dict[str, int] case.
        return size
    elif size is not None:
        # This is the int / str case.
        return int(size)
    else:
        return None

class _NamespaceDependentDefaultDict(defaultdict):

    def __init__(
        self,
        non_padded_namespaces: Iterable[str],
        padded_function: Callable[[], Any],
        non_padded_function: Callable[[], Any]) -> None:
        self.non_padded_namespaces = set(non_padded_namespaces)
        self.padded_function = padded_function
        self.non_padded_function = non_padded_function

        super(_NamespaceDependentDefaultDict, self).__init__()

    def __missing__(self, key: str):
        if any(namespace_match(pattern, key) for pattern in self.non_padded_namespaces):
            value = self.non_padded_function()
        else:
            value = self.padded_function()
        
        dict.__setitem__(self, key, value)
        return value

    def add_non_padded_namespaces(self, non_padded_namespaces: Set[str]):

        self.non_padded_namespaces.update(non_padded_namespaces)

class _Token2IdxDefaultDict(_NamespaceDependentDefaultDict):
    def __init__(self, non_padded_namespaces: Set[str], padding_token: str, oov_token: str) -> None:
        super(_Token2IdxDefaultDict, self).__init__(non_padded_namespaces, lambda: {padding_token: 0, oov_token: 1}, lambda: {})

class _Idx2TokenDefaultDict(_NamespaceDependentDefaultDict):
    def __init__(self, non_padded_namespaces: Set[str], padding_token: str, oov_token: str) -> None:
        super(_Idx2TokenDefaultDict, self).__init__(non_padded_namespaces, lambda: {0: padding_token, 1: oov_token}, lambda: {})


class Dictionary(Registrable):
    """A mapping from symbols to consecutive integers"""

    default_implementation = "default"

    def __init__(
        self,
        pad='<pad>',
        eos='</s>',
        unk='<unk>',
        bos='<s>',
        extra_special_symbols=None,
        non_padded_namespaces: Iterable[str] = DEFAULT_NON_PADDED_NAMESPACES,
        counter: Dict[str, Dict[str, int]] = None,
        min_count: Dict[str, int] = None,
        max_vocab_size: Union[int, Dict[str, int]] = None,
        pretrained_files: Optional[Dict[str, str]] = None,
        only_include_pretrained_words: bool = False,
        tokens_to_add: Dict[str, List[str]] = None,
        min_pretrained_embeddings: Dict[str, int] = None
    ) -> None:
        self.unk_word, self.pad_word, self.eos_word = unk, pad, eos
        self.symbols = []
        self.count = []
        self.indices = {}
        # dictionary indexing starts at 1 for consistency with Lua
        self.non_padded_namespaces = set(non_padded_namespaces)
        self.token2idx = _Token2IdxDefaultDict(self.non_padded_namespaces, self.pad_word, self.unk_word)
        self.idx2token = _Idx2TokenDefaultDict(self.non_padded_namespaces, self.pad_word, self.unk_word)
        self.eos_index = self.add_symbol(eos)
        self.bos_index = self.add_symbol(bos)
        if extra_special_symbols:
            for s in extra_special_symbols:
                self.add_symbol(s)
        self.nspecial = len(self.symbols)

        # Until now we've created an 'empty' dictionary, now extend it!
        self.extend(
            counter=counter,
            min_count=min_count,
            max_vocab_size=max_vocab_size,
            non_padded_namespaces=non_padded_namespaces,
            pretrained_files=pretrained_files,
            only_include_pretrained_words=only_include_pretrained_words,
            tokens_to_add=tokens_to_add,
            min_pretrained_embeddings=min_pretrained_embeddings
        )

    def __eq__(self, other):
        if isinstance(self, other.__class__):
            return self.__dict__ == other.__dict__
        return False

    def __getstate__(self):
        """
        Returns the current state of the Dictionary
        """
        state = copy.copy(self.__dict__)
        state["token2idx"] = dict(state["token2idx"])
        state["idx2token"] = dict(state["idx2token"])
        return state

    def __setstate__(self, state):
        """
        Sets the current state of this dictionary by using a state object.
        """
        state = copy.copy(self.__dict__)
        self.token2idx = _Token2IdxDefaultDict(self.non_padded_namespaces, self.pad_word, self.unk_word)
        self.idx2token = _Idx2TokenDefaultDict(self.non_padded_namespaces, self.pad_word, self.unk_word)
        self.token2idx.update(state["token2idx"])
        self.idx2token.update(state["idx2token"])

    def __getitem__(self, idx):
        return self.get_token_for_index(idx)

    def __len__(self):
        """Returns the number of symbols in the dictionary"""
        return self.get_vocab_size()

    def __contains__(self, sym):
        return self.index(sym)

    def get_vocab_size(self, namespace: str = 'tokens') -> int:
        """
        Returns the vocabulary size for the namespace.
        """
        return len(self.token2idx[namespace])

    def add_token_to_namespace(self, token: str, namespace: str = 'tokens') -> int:
        """
        Adds a new token to the dictionary for ``namespace`` and returns the index of 
        that token.
        """
        if not isinstance(token, str):
            raise ValueError("The given token is not a string!")

        if token not in self.token2idx[namespace]:
            idx = len(self.token2idx[namespace])
            self.token2idx[namespace][token] = idx
            self.idx2token[namespace][idx] = token

            return idx
        else:
            return self.token2idx[namespace][token]

    def add_tokens_to_namespace(self, tokens: List[str], namespace: str = 'tokens') -> List[int]:
        """
        Adds multiple tokens to a specific namespace.
        """
        return [self.add_token_to_namespace(token, namespace) for token in tokens]

    def get_index_to_token_vocabulary(self, namespace: str = 'tokens') -> Dict[int, str]:
        """
        Returns the whole dictionary of idx2token for a specific namespace.
        """
        return self.idx2token[namespace]

    def get_token_to_index_vocabulary(self, namespace: str = 'tokens') -> Dict[str, int]:
        """
        Returns the whole dictionary of token2idx for a specific namespace.
        """
        return self.token2idx[namespace]

    def get_token_index(self, token: str, namespace: str = 'tokens') -> int:
        """
        Returns the index of the given ``token`` within the given ``namespace```. If there is 
        no such token we are trying to return the OOV-token, if there is such a token for the 
        current namespace. Otherwise an error is thrown.
        """
        if token in self.token2idx[namespace]:
            return self.token2idx[namespace][token]
        else:
            try: 
                return self.token2idx[namespace][self.unk_word]
            except KeyError:
                LOGGER.error("Namespace: {}".format(namespace))
                LOGGER.error("\tToken: {}".format(token))
                raise

    def get_token_for_index(self, index: int, namespace: str = 'tokens') -> str:
        """
        Returns the token for a specific index.
        """
        return self.idx2token[namespace][index]

    def is_padded(self, namespace: str) -> bool:
        """
        Determines whether the dictionaries for ``namespace`` are able to pad their tokens.
        """
        return self.idx2token[namespace][0] == self.pad_word

    def index(self, sym, namespace: str = "tokens"):
        """Returns the index of the specified symbol"""
        return self.get_token_index(sym, namespace)

    def string(self, tensor, bpe_symbol=None, escape_unk=False):
        """Helper for converting a tensor of token indices to a string.

        Can optionally remove BPE symbols or escape <unk> words.
        """
        if torch.is_tensor(tensor) and tensor.dim() == 2:
            return '\n'.join(self.string(t, bpe_symbol, escape_unk) for t in tensor)

        def token_string(i):
            if i == self.unk():
                return self.unk_string(escape_unk)
            else:
                return self[i]

        # Instead of .item() one may need tolist() for multi-dimensional input
        sent = ''.join(token_string(i.item()) for i in tensor if i != self.eos())
        if bpe_symbol is not None:
            sent = (sent + ' ').replace(bpe_symbol, '').rstrip()
        return sent

    def unk_string(self, escape=False):
        """Return unknown string, optionally escaped as: <<unk>>"""
        if escape:
            return '<{}>'.format(self.unk_word)
        else:
            return self.unk_word

    def add_symbol(self, word, n=1, namespace: str = "tokens"):
        """Adds a word to the dictionary"""
        return self.add_token_to_namespace(word, namespace)

    def update(self, new_dict, namespace: str = "tokens"):
        """Updates counts from new dictionary."""
        self.add_tokens_to_namespace([word for word in new_dict.symbols], namespace)

    def finalize(self, threshold=-1, nwords=-1, padding_factor=8):
        """Sort symbols by frequency in descending order, ignoring special ones.

        Args:
            - threshold defines the minimum word count
            - nwords defines the total number of words in the final dictionary,
                including special symbols
            - padding_factor can be used to pad the dictionary size to be a
                multiple of 8, which is important on some hardware (e.g., Nvidia
                Tensor Cores).
        """
        if nwords <= 0:
            nwords = len(self)

        new_indices = dict(zip(self.symbols[:self.nspecial], range(self.nspecial)))
        new_symbols = self.symbols[:self.nspecial]
        new_count = self.count[:self.nspecial]

        c = Counter(dict(sorted(zip(self.symbols[self.nspecial:], self.count[self.nspecial:]))))
        for symbol, count in c.most_common(nwords - self.nspecial):
            if count >= threshold:
                new_indices[symbol] = len(new_symbols)
                new_symbols.append(symbol)
                new_count.append(count)
            else:
                break

        threshold_nwords = len(new_symbols)
        if padding_factor > 1:
            i = 0
            while threshold_nwords % padding_factor != 0:
                symbol = 'madeupword{:04d}'.format(i)
                new_indices[symbol] = len(new_symbols)
                new_symbols.append(symbol)
                new_count.append(0)
                i += 1
                threshold_nwords += 1

        assert len(new_symbols) % padding_factor == 0
        assert len(new_symbols) == len(new_indices)

        self.count = list(new_count)
        self.symbols = list(new_symbols)
        self.indices = new_indices

    def bos(self, namespace: str = "tokens"):
        """Helper to get index of beginning-of-sentence symbol"""
        return self.get_token_index(self.bos, namespace)

    def pad(self, namespace: str = "tokens"):
        """Helper to get index of pad symbol"""
        return self.get_token_index(self.pad_word, namespace)

    def eos(self, namespace: str = "tokens"):
        """Helper to get index of end-of-sentence symbol"""
        return self.get_token_index(self.eos_word, namespace)

    def unk(self, namespace: str = "tokens"):
        """Helper to get index of unk symbol"""
        return self.get_token_index(self.unk_word, namespace)

    @classmethod
    def load(cls, f, ignore_utf_errors=False):
        """Loads the dictionary from a text file with the format:

        ```
        <symbol0> <count0>
        <symbol1> <count1>
        ...
        ```
        """
        d = cls()
        d.add_from_file(f, ignore_utf_errors)
        return d

    def add_from_file(self, f, ignore_utf_errors=False):
        """
        Loads a pre-existing dictionary from a text file and adds its symbols
        to this instance.
        """
        if isinstance(f, str):
            try:
                if not ignore_utf_errors:
                    with open(f, 'r', encoding='utf-8') as fd:
                        self.add_from_file(fd)
                else:
                    with open(f, 'r', encoding='utf-8', errors='ignore') as fd:
                        self.add_from_file(fd)
            except FileNotFoundError as fnfe:
                raise fnfe
            except UnicodeError:
                raise Exception("Incorrect encoding detected in {}, please "
                                "rebuild the dataset".format(f))
            return

        lines = f.readlines()
        indices_start_line = self._load_meta(lines)
        for line in lines[indices_start_line:]:
            idx = line.rfind(' ')
            #if idx == -1:
            #    raise ValueError("Incorrect dictionary format, expected '<token> <cnt>'")
            word = line[:idx]
            #count = int(line[idx + 1:])
            #self.indices[word] = len(self.symbols)
            #self.symbols.append(word)
            #self.count.append(count)
            self.add_token_to_namespace(word)

    def _save(self, f, kv_iterator):
        if isinstance(f, str):
            os.makedirs(os.path.dirname(f), exist_ok=True)
            with open(f, 'w', encoding='utf-8') as fd:
                return self.save(fd)
        for k, v in kv_iterator:
            print('{} {}'.format(k, v), file=f)

    def _get_meta(self):
        return [], []

    def _load_meta(self, lines):
        return 0

    def save_to_files(self, serialization_dir: str) -> None:
        """
        Persist the Dictionary to files so it can later on be loaded again.
        Keep in mind that we have multiple namespaces, so each namespace 
        is saved within it's own file.
        """
        os.makedirs(serialization_dir, exist_ok=True)

        if os.listdir(serialization_dir):
            LOGGER.warning("The directory for the dictionary serialization '%s' is not empty.", serialization_dir)
        
        # Write all namespaces to an own file so that we know wich files we have to load (and under which namespace to load them)
        with codecs.open(os.path.join(serialization_dir, NAMESPACE_PADDING_FILE), 'w', 'utf-8') as fout:
            for namespace_name in self.non_padded_namespaces:
                print(namespace_name, file=fout)
        
        for namespace, mapping in self.idx2token.items():
            with codecs.open(os.path.join(serialization_dir, namespace + ".txt"), 'w', 'utf-8') as fout:
                num_tokens = len(mapping)
                start_idx = 1 if mapping[0] == self.pad_word else 0
                for idx in range(start_idx, num_tokens):
                    print(mapping[idx].replace("\n", "@@NEWLINE@@"), file=fout)

    def set_from_file(
        self,
        filename: str,
        is_padded: bool = False,
        oov_token: str = "<unk>",
        namespace: str = "tokens"
    ) -> None:
        """
        This method will load all namespace information from the corresponding files.
        """
        if is_padded:
            self.token2idx[namespace] = { self.pad_word: 0}
            self.idx2token[namespace] = { 0: self.pad_word}
        else:
            self.token2idx[namespace] = {}
            self.idx2token[namespace] = {}

        # Load
        with codecs.open(filename, 'r', 'utf-8') as fin:
            lines = fin.read().split('\n')

            if lines and lines[-1] == '':
                lines = lines[:-1]
            
            for idx, line in enumerate(lines):
                index = idx + 1 if is_padded else idx
                token = line.replace("@@NEWLINE@@", "\n")
                if token == "<unk>":
                    token = self.unk_word
                self.token2idx[namespace][token] = index
                self.idx2token[namespace][index] = token
            

    @classmethod
    def from_files(cls, serialization_dir: str) -> 'Dictionary':
        """
        This loads the Dictionary from files that were previously saved with ``save_to_files``.`
        """

        LOGGER.info("Loading the dictionary at '%s'", serialization_dir)

        # First load all namespace names to load and initialize
        with codecs.open(os.path.join(serialization_dir, NAMESPACE_PADDING_FILE), 'r', 'utf-8') as fin:
            non_padded_namespaces = [namespace_name.strip() for namespace_name in fin]

        # Create the dictionary according to that namespaces
        vocab = cls(non_padded_namespaces=non_padded_namespaces)

        # Now load and initialize all namespaces fmor the corresponding txt files
        for namespace_filename in os.listdir(serialization_dir):
            # Skip the namespace file
            if namespace_filename == NAMESPACE_PADDING_FILE:
                continue
            # Skip the directories '.' and '..'
            if namespace_filename.startswith('.'):
                continue

            # Now read in everything
            namespace_ = namespace_filename.replace(".txt", "")

            if any(namespace_match(pattern, namespace_) for pattern in non_padded_namespaces):
                is_padded = False
            else:
                is_padded = True
            
            filename_ = os.path.join(serialization_dir, namespace_filename)

            vocab.set_from_file(filename_, is_padded, namespace=namespace_)
        
        # Done
        return vocab

    def save(self, f):
        """Stores dictionary into a text file"""
        if isinstance(f, str):
            os.makedirs(os.path.dirname(f), exist_ok=True)
            with open(f, 'w', encoding='utf-8') as fd:
                return self.save(fd)
        #for symbol, count in zip(self.symbols[self.nspecial:], self.count[self.nspecial:]):
        for symbol in range(self.get_vocab_size()):
            #print('{} {}'.format(symbol, count), file=f)
            print('{} 10'.format(self.get_token_for_index(symbol)), file=f)

    def dummy_sentence(self, length):
        t = torch.Tensor(length).uniform_(self.nspecial + 1, len(self)).long()
        t[-1] = self.eos()
        return t

    def extend_from_instances(
        self,
        params: Params,
        instances: Iterable['adi.Instance'] = ()
    ) -> None:
        """
        Here we extend the already existing dictionary with additional instances from 
        the given datasets (instances).
        """
        min_count_ = params.pop("min_count", None)
        max_vocab_size_ = params.pop("max_vocab_size", None)

        if isinstance(max_vocab_size_, Params):
            # This is the Dict[str, int] case.
            max_vocab_size_ = max_vocab_size_.as_dict()
        elif max_vocab_size_ is not None:
            # This is the int / str case.
            max_vocab_size_ =  int(max_vocab_size_)
        else:
            max_vocab_size_ = None

        non_padded_namespaces = params.pop("non_padded_namespaces", DEFAULT_NON_PADDED_NAMESPACES)
        pretrained_files = params.pop("pretrained_files", {})
        min_pretrained_embeddings = params.pop("min_pretrained_embeddings", None)
        only_include_pretrained_words = params.pop_bool("only_include_pretrained_words", False)
        tokens_to_add = params.pop("tokens_to_add", None)

        LOGGER.info("Fitting token dictionary from dataset")

        namespace_token_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for i in Tqdm.tqdm(instances):
            i.count_vocab_items(namespace_token_counts)
        self.extend(
            counter=namespace_token_counts,
            min_count=min_count_,
            max_vocab_size=max_vocab_size_,
            non_padded_namespaces=non_padded_namespaces,
            pretrained_files=pretrained_files,
            only_include_pretrained_words=only_include_pretrained_words,
            tokens_to_add=tokens_to_add,
            min_pretrained_embeddings=min_pretrained_embeddings
        )


    def extend(
        self,
        counter: Dict[str, Dict[str, int]] = None,
        min_count: Dict[str, int] = None,
        max_vocab_size: Union[int, Dict[str, int]] = None,
        non_padded_namespaces: Iterable[str] = DEFAULT_NON_PADDED_NAMESPACES,
        pretrained_files: Optional[Dict[str, str]] = None,
        only_include_pretrained_words: bool = False,
        tokens_to_add: Dict[str, List[str]] = None,
        min_pretrained_embeddings: Dict[str, int] = None
    ) -> None:
        """ 

        TODO: Add documentation

        """
        if not isinstance(max_vocab_size, dict):
            int_max_vocab_size = max_vocab_size
            max_vocab_size = defaultdict(lambda: int_max_vocab_size)
        
        min_count = min_count or {}
        pretrained_files = pretrained_files or {}
        min_pretrained_embeddings = min_pretrained_embeddings or {}
        non_padded_namespaces = set(non_padded_namespaces)
        counter = counter or {}
        tokens_to_add = tokens_to_add or {}

        self.retained_counter = counter

        current_namespaces = {*self.token2idx}
        extension_namespaces = {*counter, *tokens_to_add}

        for namespace in current_namespaces & extension_namespaces:

            original_padded = not any(namespace_match(pattern, namespace)
                                      for pattern in self.non_padded_namespaces)
            extension_padded = not any(namespace_match(pattern, namespace)
                                       for pattern in non_padded_namespaces)
            if original_padded != extension_padded:
                raise ConfigurationError("Common namespace {} has conflicting ".format(namespace) +
                                         "setting of padded = True/False. " +
                                         "Hence extension cannot be done.")

        # 
        # Add new non-padded namespaces for extension
        self.token2idx.add_non_padded_namespaces(non_padded_namespaces)
        self.idx2token.add_non_padded_namespaces(non_padded_namespaces)
        self.non_padded_namespaces.update(non_padded_namespaces)

        for namespace in counter:
            if namespace in pretrained_files:
                pretrained_list = _read_pretrained_tokens(pretrained_files[namespace])
                min_embeddings = min_pretrained_embeddings.get(namespace, 0)
                if min_embeddings > 0:
                    tokens_old = tokens_to_add.get(namespace, [])
                    tokens_new = pretrained_list[:min_embeddings]
                    tokens_to_add[namespace] = tokens_old + tokens_new
                pretrained_set = set(pretrained_list)
            else:
                pretrained_set = None
            token_counts = list(counter[namespace].items())
            token_counts.sort(key=lambda x: x[1], reverse=True)
            try:
                max_vocab = max_vocab_size[namespace]
            except KeyError:
                max_vocab = None
            if max_vocab:
                token_counts = token_counts[:max_vocab]
            for token, count in token_counts:
                if pretrained_set is not None:
                    if only_include_pretrained_words:
                        if token in pretrained_set and count >= min_count.get(namespace, 1):
                            self.add_token_to_namespace(token, namespace)
                    elif token in pretrained_set or count >= min_count.get(namespace, 1):
                        self.add_token_to_namespace(token, namespace)
                elif count >= min_count.get(namespace, 1):
                    self.add_token_to_namespace(token, namespace)

        for namespace, tokens in tokens_to_add.items():
            for token in tokens:
                self.add_token_to_namespace(token, namespace)

    def encode_line(self, line, line_tokenizer=tokenize_line, add_if_not_exist=True,
                    consumer=None, append_eos=True, reverse_order=False):
        words = line_tokenizer(line)
        if reverse_order:
            words = list(reversed(words))
        nwords = len(words)
        ids = torch.IntTensor(nwords + 1 if append_eos else nwords)

        for i, word in enumerate(words):
            if add_if_not_exist:
                idx = self.add_symbol(word)
            else:
                idx = self.index(word)
            if consumer is not None:
                consumer(word, idx)
            ids[i] = idx
        if append_eos:
            ids[nwords] = self.eos_index
        return ids

    @staticmethod
    def _add_file_to_dictionary_single_worker(filename, tokenize, eos_word, worker_id=0, num_workers=1):
        counter = Counter()
        with open(filename, 'r', encoding='utf-8') as f:
            size = os.fstat(f.fileno()).st_size
            chunk_size = size // num_workers
            offset = worker_id * chunk_size
            end = offset + chunk_size
            f.seek(offset)
            if offset > 0:
                safe_readline(f)  # drop first incomplete line
            line = f.readline()
            while line:
                for word in tokenize(line):
                    counter.update([word])
                counter.update([eos_word])
                if f.tell() > end:
                    break
                line = f.readline()
        return counter

    @staticmethod
    def add_file_to_dictionary(filename, dict, tokenize, num_workers):
        def merge_result(counter):
            for w, c in sorted(counter.items()):
                dict.add_symbol(w, c)

        if num_workers > 1:
            pool = Pool(processes=num_workers)
            results = []
            for worker_id in range(num_workers):
                results.append(pool.apply_async(
                    Dictionary._add_file_to_dictionary_single_worker,
                    (filename, tokenize, dict.eos_word, worker_id, num_workers)
                ))
            pool.close()
            pool.join()
            for r in results:
                merge_result(r.get())
        else:
            merge_result(Dictionary._add_file_to_dictionary_single_worker(filename, tokenize, dict.eos_word))

    @classmethod
    def from_instances(
        cls,
        instances: Iterable['adi.Instance'],
        min_count: Dict[str, int] = None,
        max_vocab_size: Union[int, Dict[str, int]] = None,
        non_padded_namespaces: Iterable[str] = DEFAULT_NON_PADDED_NAMESPACES,
        pretrained_files: Optional[Dict[str, int]] = None,
        only_include_pretrained_words: bool = False,
        tokens_to_add: Dict[str, List[str]] = None,
        min_pretrained_embeddings: Dict[str, int] = None
    ) -> 'Vocabulary':
        """
        This static method will construct a dictionary from a given set of Instances and some 
        additional parameters. We count all of the dictionary items in the instances.
        """
        LOGGER.info("fitting token dictionary from the given dataset.")
        namespace_token_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for instance in Tqdm.tqdm(instances):
            instance.count_vocab_items(namespace_token_counts)

        # Finally, create the dictionary instance
        return cls()

    @classmethod
    def from_params(cls, params: Params, instances: Iterable['adi.Instance'] = None):
        """
        """
        dictionary_type = params.pop("type", None)
        if dictionary_type is not None:
            return cls.by_name(dictionary_type).from_params(params=params, instances=instances)

        # Should we extend the dictionary
        extend = params.pop("extend", False)
        dictionary_path = params.pop("directory_path", None)

        if not dictionary_path and not instances:
            raise ConfigurationError("you must either provide a directory_path inside the parameters or a dataset to build a dictionary from")

        if extend and not instances:
            raise ConfigurationError("'extend' is activated, but there are no instances to pass through")
        if extend and not dictionary_path:
            raise ConfigurationError("'entend' is activated, but there is no 'directory_path' to extend from.")

        if dictionary_path and instances:
            if extend:
                LOGGER.info("loading the dictionary from files and extending it with a dataset.")
            else:
                LOGGER.info("loading the dictionary from files instead of a dataset")

        # Enough parameter evaluation, now let's finally create and initialize the data
        if dictionary_path:
            vocab = cls.from_files(dictionary_path)
            if not extend:
                return vocab

        if extend:
            vocab.extend_from_instances(params, instances=instances)
            return vocab

        # There is no dictionary path given and we should not extend, so we have to create the 
        # vocabulary from a dataset
        min_count = params.pop("min_count", None, keep_as_dict=True)
        max_vocab_size = pop_max_vocab_size(params)
        non_padded_namespaces = params.pop("non_padded_namespaces", DEFAULT_NON_PADDED_NAMESPACES)
        pretrained_files = params.pop("pretrained_files", {}, keep_as_dict=True)
        min_pretrained_embeddings = params.pop("min_pretrained_embeddings", None)
        only_include_pretrained_words = params.pop_bool("only_include_pretrained_words", False)
        tokens_to_add = params.pop("tokens_to_add", None)

        return cls.from_instances(
            instances=instances,
            min_count=min_count,
            max_vocab_size=max_vocab_size,
            non_padded_namespaces=non_padded_namespaces,
            pretrained_files=pretrained_files,
            only_include_pretrained_words=only_include_pretrained_words,
            tokens_to_add=tokens_to_add,
            min_pretrained_embeddings=min_pretrained_embeddings
        )

class TruncatedDictionary(object):

    def __init__(self, wrapped_dict, length):
        self.__class__ = type(
            wrapped_dict.__class__.__name__,
            (self.__class__, wrapped_dict.__class__),
            {}
        )
        self.__dict__ = wrapped_dict.__dict__
        self.wrapped_dict = wrapped_dict
        self.length = min(len(self.wrapped_dict), length)

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        if i < self.length:
            return self.wrapped_dict[i]
        return self.wrapped_dict.unk()
