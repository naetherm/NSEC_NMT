# -*- coding: utf-8 -*-
'''
Copyright 2019, University of Freiburg.
Chair of Algorithms and Data Structures.
Markus Näther <naetherm@informatik.uni-freiburg.de>
'''

"""
Translate raw text with a trained model. Batches data on-the-fly.
"""

from collections import namedtuple
import numpy as np
import sys
import glob
import regex as re
import json
import fileinput
import copy

import torch

from fairseq import data, options, tasks, tokenizer, utils, checkpoint_utils
from fairseq.data import encoders
from fairseq.sequence_generator import SequenceGenerator


Batch = namedtuple('Batch', 'ids src_tokens src_lengths')
Translation = namedtuple('Translation', 'src_str hypos pos_scores alignments')

class SourceInternalArticle(object):

  def __init__(self):
    self.sentences = None

def build_article_information(input):
  json_ = input # json.loads(input)
  articles = set()
  for t in json_["tokens"]:
    nums_ = re.findall('\d+', t['id'], re.UNICODE)
    articles.add(int(nums_[0]))
  # count sentences
  num_articles = len(articles)
  #print(f"detected {num_articles} in source file, collect number of sentences")
  num_sentences = [set() for _ in range(num_articles)]
  for t in json_["tokens"]:
    nums_ = re.findall('\d+', t['id'], re.UNICODE)
    num_sentences[int(nums_[0])].add(int(nums_[1]))

  results = [SourceInternalArticle() for _ in range(num_articles)]
  for aidx in range(num_articles):
    #print(f"detected {len(num_sentences[aidx])} sentences for article {aidx}")
    #for sidx in range(len(num_sentences[aidx])):
    results[aidx].sentences = ["" for _ in range(len(num_sentences[aidx]))]

  for t in json_["tokens"]:
    nums_ = re.findall('\d+', t['id'], re.UNICODE)
    results[int(nums_[0])].sentences[int(nums_[1])] += t['token']
    if ((t['space'] == True) or (t['space'] == 'true')):
      results[int(nums_[0])].sentences[int(nums_[1])] += ' '

  return copy.deepcopy(results)

def call_regex(src):
    '''
    One place for the regex.
    '''
    tokens = re.findall(r"(?:\d+,\d+)|(?:[\w'\u0080-\u9999]+(?:[-]+[\w'\u0080-\u9999]+)+)|(?:[\w\u0080-\u9999]+(?:[']+[\w\u0080-\u9999]+)+)|\b[_]|(?:[_]*[\w\u0080-\u9999]+(?=_\b))|(?:[\w\u0080-\u9999]+)|[^\w\s\p{Z}]", src, re.UNICODE)

    spaces = []
    char_counter = 0
    for t in tokens:
        char_counter += len(t)
        if char_counter >= len(src):
            spaces.append(False)
        else:
            if src[char_counter] == ' ':
                spaces.append(True)
                char_counter += 1
            else:
                spaces.append(False)

    return tokens, spaces

def generate_token_information(aidx, sidx, tidx, token, suggestions, space, add_comma, proposed_type=None):
  if token == "\\":
    print("WARNING: %d %d %d %s" % (aidx, sidx, tidx, token))
  result = "  {"

  if isinstance(tidx, list):
    result += "\"id\": \"a" + str(aidx) + ".s" + str(sidx) + ".w" + str(tidx[0]) + "-a" + str(aidx) + ".s" + str(sidx) + ".w" + str(tidx[1]) + "\", "
  else:
    result += "\"id\": \"a" + str(aidx) + ".s" + str(sidx) + ".w" + str(tidx) + "\", "
  if proposed_type != None:
    result += "\"type\": \"" + proposed_type + "\", "
  result += "\"token\": \"" + token.replace("\\", "\\\\").replace("\"", "\\\"") + "\", "

  result += "\"suggestions\": ["

  for idx, suggestion in enumerate(suggestions):
    result += "\"" + suggestion.replace("\\", "\\\\").replace("\"", "\\\\\"") + "\""
    if idx < (len(suggestions)-1):
      result += ", "

  result += "], \"space\": "

  if space:
    result += "true"
  else:
    result += "false"

  if add_comma:
    result += "},\n"
  else:
    result += "}"

  return result

def buffered_read(buffer_size):
    buffer = []
    for src_str in sys.stdin:
        buffer.append(src_str.strip())
        if len(buffer) >= buffer_size:
            yield buffer
            buffer = []

    if len(buffer) > 0:
        yield buffer


def make_batches(lines, args, task, max_positions, encode_fn):
    tokens = [
        task.source_dictionary.encode_line(
            encode_fn(src_str), add_if_not_exist=False
        ).long()
        for src_str in lines
    ]
    lengths = torch.LongTensor([t.numel() for t in tokens])
    itr = task.get_batch_iterator(
        dataset=task.build_dataset_for_inference(tokens, lengths),
        max_tokens=args.max_tokens,
        max_sentences=args.max_sentences,
        max_positions=max_positions,
    ).next_epoch_itr(shuffle=False)
    for batch in itr:
        yield Batch(
            ids=batch['id'],
            src_tokens=batch['net_input']['src_tokens'],
            src_lengths=batch['net_input']['src_lengths'],
        )


def main(args):
    # Import the user modules
    utils.import_user_module(args)

    if args.buffer_size < 1:
        args.buffer_size = 1
    if args.max_tokens is None and args.max_sentences is None:
        args.max_sentences = 1

    assert not args.sampling or args.nbest == args.beam, \
        '--sampling requires --nbest to be equal to --beam'
    assert not args.max_sentences or args.max_sentences <= args.buffer_size, \
        '--max-sentences/--batch-size cannot be larger than --buffer-size'

    print(args)

    use_cuda = torch.cuda.is_available() and not args.cpu

    # Setup task, e.g., translation
    task = tasks.setup_task(args)

    # Load ensemble
    print('| loading model(s) from {}'.format(args.path))
    model_paths = args.path.split(':')
    # This line is deprecated
    models, _model_args = checkpoint_utils.load_model_ensemble(
        args.path.split(':'),
        arg_overrides=eval(args.model_overrides),
        task=task
    )

    # Set dictionaries
    src_dict = task.source_dictionary
    tgt_dict = task.target_dictionary

    # Optimize ensemble for generation
    for model in models:
        model.make_generation_fast_(
            beamable_mm_beam_size=None if args.no_beamable_mm else args.beam,
            need_attn=args.print_alignment,
        )
        if args.fp16:
            model.half()
        if use_cuda:
            model.cuda()

    # Initialize generator
    translator = task.build_generator(args)

    tokenizer = encoders.build_tokenizer(args)
    bpe = encoders.build_bpe(args)

    def encode_fn(x):
        if tokenizer is not None:
            x = tokenizer.encode(x)
        if bpe is not None:
            x = bpe.encode(x)
        return x

    def decode_fn(x):
        if bpe is not None:
            x = bpe.decode(x)
        if tokenizer is not None:
            x = tokenizer.decode(x)
        return x

    # Load alignment dictionary for unknown word replacement
    # (None if no unknown word replacement, empty if no path to align dictionary)
    align_dict = utils.load_align_dict(args.replace_unk)

    max_positions = utils.resolve_max_positions(
        task.max_positions(),
        *[model.max_positions() for model in models]
    )

    def make_result(src_str, hypos):
        result = Translation(
            src_str='O\t{}'.format(src_str),
            hypos=[],
            pos_scores=[],
            alignments=[],
        )

        # Process top predictions
        for hypo in hypos[:min(len(hypos), args.nbest)]:
            hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
                hypo_tokens=hypo['tokens'].int().cpu(),
                src_str=src_str,
                alignment=hypo['alignment'].int().cpu() if hypo['alignment'] is not None else None,
                align_dict=align_dict,
                tgt_dict=tgt_dict,
                remove_bpe=args.remove_bpe,
            )
            result.hypos.append('{}'.format(hypo_str))
            result.pos_scores.append('P\t{}'.format(
                ''.join(map(
                    lambda x: '{:.4f}'.format(x),
                    hypo['positional_scores'].tolist(),
                ))
            ))
            result.alignments.append(
                'A\t{}'.format(''.join(map(lambda x: str(utils.item(x)), alignment)))
                if args.print_alignment else None
            )
        return result

    def process_batch(batch):
        tokens = batch.tokens
        lengths = batch.lengths

        if use_cuda:
            tokens = tokens.cuda()
            lengths = lengths.cuda()

        encoder_input = {'src_tokens': tokens, 'src_lengths': lengths}
        translations = translator.generate(
            encoder_input,
            maxlen=int(args.max_len_a * tokens.size(1) + args.max_len_b),
        )

        return [make_result(batch.srcs[i], t) for i, t in enumerate(translations)]

    if args.buffer_size > 1:
        print('| Sentence buffer size:', args.buffer_size)
    filename = '/home/naetherm/repositories/data/benchmark/smallish_benchmark_source.json'
    outfile_name = './prediction.json' 
    outfile = open(outfile_name, 'w')
    result_content = "{ \"predictions\": [\n"
    content = None
    with open(filename, 'r', encoding='utf-8') as fin:
        content = json.load(fin)


    articles = build_article_information(content)
    
    start_id = 0
    for aidx, article in enumerate(articles):
        for sidx, inputs in enumerate(article.sentences):
            #print("Currently processing sentence {} of {} ...".format(sidx, len(sentences)))
            inputs = [inputs.replace("\n", "")]
            results = []
            for batch in make_batches(inputs, args, task, max_positions, encode_fn):
                src_tokens = batch.src_tokens
                src_lengths = batch.src_lengths
                if use_cuda:
                    src_tokens = src_tokens.cuda()
                    src_lengths = src_lengths.cuda()

                sample = {
                    'net_input': {
                        'src_tokens': src_tokens,
                        'src_lengths': src_lengths,
                    },
                }
                translations = task.inference_step(translator, models, sample)
                for i, (id, hypos) in enumerate(zip(batch.ids.tolist(), translations)):
                    src_tokens_i = utils.strip_pad(src_tokens[i], tgt_dict.pad())
                    results.append((start_id + id, src_tokens_i, hypos))

            # sort output to match input order
            for id, src_tokens, hypos in sorted(results, key=lambda x: x[0]):
                if src_dict is not None:
                    src_str = src_dict.string(src_tokens, args.remove_bpe)

                # Process top predictions
                for hypo in hypos[:min(len(hypos), args.nbest)]:
                    hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
                        hypo_tokens=hypo['tokens'].int().cpu(),
                        src_str=src_str,
                        alignment=hypo['alignment'].int().cpu() if hypo['alignment'] is not None else None,
                        align_dict=align_dict,
                        tgt_dict=tgt_dict,
                        remove_bpe=args.remove_bpe,
                    )
                    hypo_str = decode_fn(hypo_str)
                    tokens, spaces = call_regex(hypo_str)
                    for tidx, token in enumerate(tokens):
                        result_content += generate_token_information(
                            aidx, 
                            sidx, 
                            tidx, 
                            token, 
                            [], 
                            spaces[tidx], 
                            not ( (tidx == (len(tokens)-1)) and (sidx == (len(article.sentences)-1)) and (aidx == (len(articles)-1)))
                        )
                if aidx < (len(articles) - 1) and sidx < (len(article.sentences) - 1):
                  result_content += ",\n"
                

    result_content += "  ]\n}"
    outfile.write(result_content)
    outfile.close()

    #inp_file.close()


if __name__ == '__main__':
    parser = options.get_generation_parser(interactive=True)
    args = options.parse_args_and_arch(parser)
    main(args)
