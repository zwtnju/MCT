import argparse

from run_pre_train import pre_train
from dataset import CodeMAEDataset
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type: e.g. roberta")
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained results: e.g. roberta-base")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the results predictions and checkpoints will be written.")
    parser.add_argument("--load_model_path", default=None, type=str,
                        help="Path to trained results: Should contain the .bin files")
    # Other parameters
    parser.add_argument("--train_filename", default="", type=str,
                        help="The train filename. Should contain the .jsonl files for this task.")

    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")

    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased results.")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")

    parser.add_argument("--train_batch_size", default=32, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--beam_size", default=1, type=int,
                        help="beam size for beam search")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--eval_steps", default=-1, type=int,
                        help="")
    parser.add_argument("--train_steps", default=-1, type=int,
                        help="")
    parser.add_argument("--train_epochs", default=1.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    # additional parameters
    parser.add_argument('--max_seq_len', type=int, default=256,
                        help="max sequence length of code")
    parser.add_argument('--max_nl_len', type=int, default=38,
                        help="max nature language description of code")
    parser.add_argument('--ignore_index', type=int, default=-100,
                        help="ignore index when calculate loss")
    parser.add_argument('--special_split', type=str, default='Ġ',
                        help="the split symbol of tokenizer")
    parser.add_argument('--max_checkpoints', type=int, default=2,
                        help="max checkpoint number")

    parser.add_argument('--mask_type', type=str, default='func',
                        help="mask type of code, including iden, const, var, func, str")
    parser.add_argument('--total_mask_type', type=str, default='',
                        help="mask type of code")
    # print arguments
    args = parser.parse_args()

    pre_train(args)
