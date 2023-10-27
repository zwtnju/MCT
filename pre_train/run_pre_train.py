import logging
import os
import pickle

import torch
from transformers import (RobertaConfig, RobertaTokenizer, RobertaForMaskedLM,
                          PLBartConfig, PLBartTokenizer, PLBartForConditionalGeneration,
                          T5Config, T5ForConditionalGeneration, PLBartModel)
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, IntervalStrategy

from data_collator_bpe import data_collator_fn
from dataset import CodeMAEDataset

MODEL_CLASSES = {'codebert': (RobertaConfig, RobertaForMaskedLM, RobertaTokenizer),
                 'codet5': (T5Config, T5ForConditionalGeneration, RobertaTokenizer),
                 'plbart': (PLBartConfig, PLBartForConditionalGeneration, PLBartTokenizer)}

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def pre_train(args):
    # dataset
    train_set = pickle.load(open("train_data.pkl", "rb"))

    # results
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
                                                do_lower_case=args.do_lower_case)

    if args.load_model_path is not None:
        logger.info("reload results from {}".format(args.load_model_path))
        model = model_class.from_pretrained(args.load_model_path, config=config)
    else:
        model = model_class.from_pretrained(args.model_name_or_path, config=config)

    if os.path.exists(args.output_dir) is False:
        os.makedirs(args.output_dir)

    # trainer
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.train_batch_size,
        do_train=True,
        num_train_epochs=args.train_epochs,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        save_total_limit=args.max_checkpoints,
        logging_dir=args.output_dir,
        logging_strategy=IntervalStrategy.STEPS,
        load_best_model_at_end=True,
        evaluation_strategy=IntervalStrategy.NO,
        save_strategy=IntervalStrategy.NO,
        seed=args.seed,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_set,
        tokenizer=tokenizer,
        data_collator=lambda batch: data_collator_fn(batch, tokenizer, args),
    )

    trainer.train()
    trainer.save_model()