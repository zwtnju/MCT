from __future__ import absolute_import

import copy
import math
import os
import sys
import pickle
import torch
import json
import random
import logging
import argparse
import numpy as np
from io import open
from itertools import cycle
import torch.nn as nn

from data_collator_bpe import mask_constant, mask_random, mask_string, mask_nonrandom
from dataset import CodeMAEDataset
from tqdm import tqdm, trange
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaModel, RobertaTokenizer, RobertaForMaskedLM, PLBartConfig,
                          PLBartTokenizer, BartConfig, T5Config,
                          T5ForConditionalGeneration, BartForConditionalGeneration, BartTokenizer, T5Tokenizer,
                          PLBartForConditionalGeneration, PLBartModel, T5Model, AutoModelForSeq2SeqLM, AutoTokenizer,
                          AutoConfig)

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


MODEL_CLASSES = {
    'roberta': (RobertaConfig, RobertaForMaskedLM, RobertaTokenizer),
    'codet5': (T5Config, T5ForConditionalGeneration, RobertaTokenizer),
    'plbart': (PLBartConfig, PLBartForConditionalGeneration, PLBartTokenizer)}


class InputFeatures(object):
    """A single training/test features for a example."""

    def __init__(self,
                 example_id,
                 source_ids,
                 target_ids,
                 source_mask,
                 source_token_type_id,
                 ):
        self.example_id = example_id
        self.source_ids = source_ids
        self.target_ids = target_ids
        self.source_mask = source_mask
        self.source_token_type_id = source_token_type_id


def convert_examples_to_features(examples, tokenizer, args, stage=None):
    features = []

    for example_index, example_data in tqdm(enumerate(examples), total=len(examples)):

        doc, lang, code_mask_str, code_strings, \
        code_mask_var, code_mask_func_call, code_mask_identifier, \
        code_variables, codes_func_calls, codes_identifiers = example_data

        for special_token in tokenizer.all_special_tokens:
            code_mask_str = code_mask_str.replace(special_token, "")
            code_mask_var = code_mask_var.replace(special_token, "")
            code_mask_func_call = code_mask_func_call.replace(special_token, "")
            code_mask_identifier = code_mask_identifier.replace(special_token, "")

        if args.mask_type == 'const':
            source_ids, source_tokens, source_token_type_id, source_mask, target_tokens, target_ids = \
                mask_constant(doc, lang, code_mask_str, code_strings, code_mask_identifier, codes_identifiers, tokenizer,
                              args)
        elif args.mask_type == 'str':
            source_ids, source_tokens, source_token_type_id, source_mask, target_tokens, target_ids = \
                mask_string(doc, lang, code_mask_str, code_strings, tokenizer, args)
        elif args.mask_type == 'var':
            source_ids, source_tokens, source_token_type_id, source_mask, target_tokens, target_ids = \
                mask_nonrandom(doc, lang, code_mask_str, code_strings, code_mask_var, code_variables, tokenizer, args)
        elif args.mask_type == 'func':
            source_ids, source_tokens, source_token_type_id, source_mask, target_tokens, target_ids = \
                mask_nonrandom(doc, lang, code_mask_str, code_strings, code_mask_func_call, codes_func_calls, tokenizer,
                               args)
        elif args.mask_type == 'iden':
            source_ids, source_tokens, source_token_type_id, source_mask, target_tokens, target_ids = \
                mask_nonrandom(doc, lang, code_mask_str, code_strings, code_mask_identifier, codes_identifiers,
                               tokenizer,
                               args)
        else:
            assert args.mask_type == 'rand', f"error type"
            source_ids, source_tokens, source_token_type_id, source_mask, target_tokens, target_ids = \
                mask_random(doc, lang, code_mask_str, code_strings, tokenizer, args)
        if source_ids:
            if example_index < 5:
                if stage == 'train':
                    logger.info("*** Example ***")
                    logger.info("idx: {}".format(example_index))

                    logger.info("source_tokens: {}".format([x.replace('\u0120', '_') for x in source_tokens]))
                    logger.info("source_ids: {}".format(' '.join(map(str, source_ids))))
                    logger.info("source_mask: {}".format(' '.join(map(str, source_mask))))
                    logger.info("source_token_type_id: {}".format(' '.join(map(str, source_token_type_id))))

                    target_tokens_rm_null = []
                    for i in target_tokens:
                        if i:
                            target_tokens_rm_null.append(i)

                    logger.info("target_tokens: {}".format([x.replace('\u0120', '_') for x in target_tokens_rm_null]))
                    logger.info("target_ids: {}".format(' '.join(map(str, target_ids))))

            features.append(
                InputFeatures(
                    example_index,
                    source_ids,
                    target_ids,
                    source_mask,
                    source_token_type_id,
                )
            )
    return features


def generate_data():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type: e.g. roberta")
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained results: e.g. roberta-base")

    # Other parameters

    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")

    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased results.")

    # additional parameters
    parser.add_argument('--max_seq_len', type=int, default=256,
                        help="max sequence length of code")
    parser.add_argument('--max_nl_len', type=int, default=38,
                        help="max nature language description of code")
    parser.add_argument('--ignore_index', type=int, default=-100,
                        help="ignore index when calculate loss")
    parser.add_argument('--mask_type', type=str, default="")
    parser.add_argument('--special_split', type=str, default='Ġ')

    # print arguments
    args = parser.parse_args()
    logger.info(args)

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
                                                do_lower_case=args.do_lower_case)
    train_examples = pickle.load(open("train_data.pkl", "rb"))
    # train_examples = train_examples.subset(0.001)
    # for mask_type in ['iden']:  # ['const', 'var', 'func', 'rand', 'iden', 'str']:
    mask_type = args.mask_type
    logger.info(args.model_type + " model generates " + mask_type + " dataset")
    train_features = convert_examples_to_features(train_examples, tokenizer, args, stage='train')
    all_source_ids = torch.tensor([f.source_ids for f in train_features], dtype=torch.long)
    all_source_mask = torch.tensor([f.source_mask for f in train_features], dtype=torch.long)
    all_source_token_type_id = torch.tensor([f.source_token_type_id for f in train_features], dtype=torch.long)
    all_target_ids = torch.tensor([f.target_ids for f in train_features], dtype=torch.long)
    train_data = TensorDataset(all_source_ids, all_source_mask, all_source_token_type_id, all_target_ids)

    save_file_str = args.model_name + '_' + mask_type + '.pt'
    torch.save(train_data, save_file_str)
    logger.info("generate data finished")


if __name__ == "__main__":
    generate_data()
