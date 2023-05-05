import torch
from torch import nn

# -------------------------------dataset args-------------------------------#

repo = 'repo'  # the owner/repo
path = 'path'  # the full path to the original file
func_name = 'func_name'  # the function or method name
orig_str = 'original_string'  # the raw string before tokenization or parsing
# Lang = 'language'  # the programming language
code = 'code'  # the part of the original_string that is code
code_t = 'code_tokens'  # tokenized version of code
doc = 'docstring'  # the top-level comment or docstring, if it exists in the original string
doc_t = 'docstring_tokens'  # tokenized version of docstring
# Sha = 'sha'  # this field is not being used
# Part = 'partition'  # a flag indicating what partition this datum belongs to of {train, valid, test, etc.} This is
# not used by the results. Instead we rely on directory structure to denote the partition of the data.
url = 'url'  # the url for the code snippet including the line numbers

go = 'go'
java = 'java'
javascript = 'javascript'
php = 'php'
python = 'python'
ruby = 'ruby'
langs = [go, java, javascript, php, python, ruby]

dataset_dir = "./dataset/pre_train/"
go_dir = f"{dataset_dir}{go}/"
java_dir = f"{dataset_dir}{java}/"
javascript_dir = f"{dataset_dir}{javascript}/"
php_dir = f"{dataset_dir}{php}/"
python_dir = f"{dataset_dir}{python}/"
ruby_dir = f"{dataset_dir}{ruby}/"
err_dir = f"{dataset_dir}err_codes/"
go_err_dir = f"{err_dir}{go}"
java_err_dir = f"{err_dir}{java}"
javascript_err_dir = f"{err_dir}{javascript}"
php_err_dir = f"{err_dir}{php}"
python_err_dir = f"{err_dir}{python}"
ruby_err_dir = f"{err_dir}{ruby}"

doc_file_name = 'doc'
code_file_name = 'code'
url_file_name = 'url'
err_code_file_name = 'err_code'
err_url_file_name = 'err_url'

go_ast_file = './tree_sitter_tools/my-languages.so'
java_ast_file = './tree_sitter_tools/my-languages.so'
javascript_ast_file = './tree_sitter_tools/my-languages.so'
php_ast_file = './tree_sitter_tools/my-languages.so'
python_ast_file = './tree_sitter_tools/my-languages.so'
ruby_ast_file = './tree_sitter_tools/my-languages.so'

downstream_tasks_dataset_dir = "../../dataset/fine_tune/"
bug_fix_dir = downstream_tasks_dataset_dir + 'bug_fix/'
clone_detection_dir = downstream_tasks_dataset_dir + 'clone_detection/big_clone_bench/'
code_completion_dir = downstream_tasks_dataset_dir + 'code_completion/'
code_qa_dir = downstream_tasks_dataset_dir + 'code_qa/fdm/'
code_retrieval_dir = downstream_tasks_dataset_dir + 'code_retrieval/'
code_translation_dir = downstream_tasks_dataset_dir + 'code_translation/'
defect_detection_dir = downstream_tasks_dataset_dir + 'defect_detection/devign/'
exception_type_dir = downstream_tasks_dataset_dir + 'exception_type/exception/'
code_summarization_dir = downstream_tasks_dataset_dir + 'code_summarization/CodeSearchNet'

valid_code_file_name = 'valid_code'
valid_doc_file_name = 'valid_doc'
mask_var_str_code_file_name = 'mask_var_str_code'
code_variables_file_name = 'code_variables'
mask_function_call_str_code_file_name = 'mask_function_call_str_code'
code_function_calls_file_name = 'code_function_calls'
mask_identifier_str_code_file_name = 'mask_identifier_str_code'
codes_identifiers_file_name = 'codes_identifiers'
mask_str_code_file_name = 'mask_str_code'
code_strings_file_name = 'code_strings'
err_useful_code_file_name = 'err_useful_code'
err_useful_url_file_name = 'err_useful_url'

# -------------------------------vocab args-------------------------------#
vocab_size = 80000

unk_token = "[UNK]"
cls_token = "[CLS]"
sep_token = "[SEP]"
eos_token = "[EOS]"
pad_token = "[PAD]"
mask_token = "[MASK]"

unk_id = 0
cls_id = 1
sep_id = 2
eos_id = 3
pad_id = 4
mask_id = 5

special_tokens = [unk_token, cls_token, sep_token, eos_token, pad_token, mask_token]

doc_empty_token = '[NONE]'
code_empty_token = ''
string_replaced_token = '[NCAST]'
var_replaced_token = '[TZLG]'
string_underline_token = '_'
new_line_replaced_token = ' '
tab_replaced_token = ' '

# word_split_prefix = '##'
special_double_symbol_list = ['*', ':', '"', '&', '=', '/', '%', '|', '+', '-']

other_tokens = [doc_empty_token, string_replaced_token, var_replaced_token, ]

# add languages as a prompt
additional_tokens = langs + other_tokens

# -------------------------------results args-------------------------------#
subset_ratio = .00001

# small
embed_dim = 512  # embedding dimension
num_heads = 8  # number of heads in multi-head attention
num_layers = 6  # number of transformer layers
max_seq_len = 256  # set max length for a code sequence
max_nl_len = round(256 * 0.15)

# base
# embed_dim = 768
# num_heads = 12
# num_layers = 12
# max_seq_len = 514
# max_nl_len = round(max_seq_len * 0.15)
feedforward_dim = 4 * embed_dim  # the dimension of mlp layer

# decoder_embed_dim = 8
# decoder_num_heads = 2
# decoder_num_layers = 1
# decoder_feedforward_dim = 4 * decoder_embed_dim

dropout = 0.1  # dropout probability
max_source_length = 256
max_target_length = 128
beam_size = 5

weight_decay = 0.0
learning_rate = 5e-5
lr = 5e-5  # learning rate
adam_epsilon = 1e-8
gradient_accumulation_steps = 2

# understanding tasks
batch_size = 32
epochs = 2
# generation tasks
train_batch_size = 8
num_train_epochs = 50

steps = 20000

eval_batch_size = 32

max_checkpoints = 10
random_seed = 42
ignore_index = -100

warmup_steps = 0
patience = 5

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
n_gpu = 4

model_save_dir = './results/'
tokenizer_file_name = 'tokenizer.json'
vocab_file_name = 'vocab.txt'

pre_train_file_dir = model_save_dir + 'pre_train/'
backup_vocab_file = dataset_dir + tokenizer_file_name

# different pre_train models
pre_train_mask_identifier_dir = pre_train_file_dir + 'task_mask_identifier/'
pre_train_mask_constant_dir = pre_train_file_dir + 'task_mask_constant/'
pre_train_mask_var_dir = pre_train_file_dir + 'task_mask_var/'
pre_train_mask_func_call_dir = pre_train_file_dir + 'task_mask_func_call/'
pre_train_mask_rand_dir = pre_train_file_dir + 'task_mask_rand/'
pre_train_mask_str_dir = pre_train_file_dir + 'task_mask_str/'

pre_train_mask_identifier_constant_dir = pre_train_file_dir + 'task_mask_identifier_constant/'
pre_train_mask_identifier_rand_dir = pre_train_file_dir + 'task_mask_identifier_rand/'
pre_train_mask_identifier_str_dir = pre_train_file_dir + 'task_mask_identifier_str/'
pre_train_mask_identifier_constant_rand_dir = pre_train_file_dir + 'task_mask_identifier_constant_rand/'
pre_train_mask_identifier_constant_str_dir = pre_train_file_dir + 'task_mask_identifier_constant_str/'
pre_train_mask_identifier_rand_str_dir = pre_train_file_dir + 'task_mask_identifier_rand_str/'
pre_train_mask_identifier_constant_rand_str_dir = pre_train_file_dir + 'task_mask_identifier_constant_rand_str/'

pre_train_mask_constant_var_dir = pre_train_file_dir + 'task_mask_constant_var/'
pre_train_mask_constant_func_dir = pre_train_file_dir + 'task_mask_constant_func/'
pre_train_mask_constant_rand_dir = pre_train_file_dir + 'task_mask_constant_rand/'
pre_train_mask_constant_str_dir = pre_train_file_dir + 'task_mask_constant_str/'
pre_train_mask_constant_var_func_dir = pre_train_file_dir + 'task_mask_constant_var_func/'
pre_train_mask_constant_var_rand_dir = pre_train_file_dir + 'task_mask_constant_var_rand/'
pre_train_mask_constant_var_str_dir = pre_train_file_dir + 'task_mask_constant_var_str/'
pre_train_mask_constant_func_rand_dir = pre_train_file_dir + 'task_mask_constant_func_rand/'
pre_train_mask_constant_func_str_dir = pre_train_file_dir + 'task_mask_constant_func_str/'
pre_train_mask_constant_rand_str_dir = pre_train_file_dir + 'task_mask_constant_rand_str/'
pre_train_mask_constant_var_func_rand_dir = pre_train_file_dir + 'task_mask_constant_var_func_rand/'
pre_train_mask_constant_var_func_str_dir = pre_train_file_dir + 'task_mask_constant_var_func_str/'
pre_train_mask_constant_var_rand_str_dir = pre_train_file_dir + 'task_mask_constant_var_rand_str/'
pre_train_mask_constant_func_rand_str_dir = pre_train_file_dir + 'task_mask_constant_func_rand_str/'
pre_train_mask_constant_var_func_rand_str_dir = pre_train_file_dir + 'task_mask_constant_var_func_rand_str/'

pre_train_mask_var_func_dir = pre_train_file_dir + 'task_mask_var_func/'
pre_train_mask_var_rand_dir = pre_train_file_dir + 'task_mask_var_rand/'
pre_train_mask_var_str_dir = pre_train_file_dir + 'task_mask_var_str/'
pre_train_mask_var_func_rand_dir = pre_train_file_dir + 'task_mask_var_func_rand/'
pre_train_mask_var_func_str_dir = pre_train_file_dir + 'task_mask_var_func_str/'
pre_train_mask_var_rand_str_dir = pre_train_file_dir + 'task_mask_var_rand_str/'
pre_train_mask_var_func_rand_str_dir = pre_train_file_dir + 'task_mask_var_func_rand_str/'

pre_train_mask_func_var_dir = pre_train_file_dir + 'task_mask_func_var/'
pre_train_mask_func_rand_dir = pre_train_file_dir + 'task_mask_func_rand/'
pre_train_mask_func_str_dir = pre_train_file_dir + 'task_mask_func_str/'
pre_train_mask_func_rand_str_dir = pre_train_file_dir + 'task_mask_func_rand_str/'

pre_train_mask_rand_str_dir = pre_train_file_dir + 'task_mask_rand_str/'

# code-bert based pre-train results save dir
bert_file_dir = model_save_dir + 'bert/'
bert_mask_func_dir = bert_file_dir + 'mask_func/'
bert_mask_cons_dir = bert_file_dir + 'mask_cons/'

# mask type
mask_var_tokens = 'mask_var'
mask_func_tokens = 'mask_func_call'
mask_ids_tokens = 'mask_identifier'
mask_str_tokens = 'mask_string'
mask_rand_tokens = 'mask_rand'
mask_constant_tokens = 'mask_constant'
mask_types = [mask_var_tokens, mask_func_tokens, mask_ids_tokens, mask_str_tokens, mask_rand_tokens]

# -------------------------------downstream_tasks args-------------------------------#
downstream_tasks_model_save_dir = '../.' + model_save_dir
downstream_tasks_vocab_file = '../.' + backup_vocab_file

# downstream_tasks_pre_train_file_dir = '../.' + pre_train_file_dir

downstream_tasks_pre_train_mask_identifier_dir = '../.' + pre_train_mask_identifier_dir
downstream_tasks_pre_train_mask_constant_dir = '../.' + pre_train_mask_constant_dir
downstream_tasks_pre_train_mask_var_dir = '../.' + pre_train_mask_var_dir
downstream_tasks_pre_train_mask_func_call_dir = '../.' + pre_train_mask_func_call_dir
downstream_tasks_pre_train_mask_rand_dir = '../.' + pre_train_mask_rand_dir
downstream_tasks_pre_train_mask_str_dir = '../.' + pre_train_mask_str_dir

downstream_tasks_pre_train_mask_identifier_constant_dir = '../.' + pre_train_mask_identifier_constant_dir
downstream_tasks_pre_train_mask_identifier_rand_dir = '../.' + pre_train_mask_identifier_rand_dir
downstream_tasks_pre_train_mask_identifier_str_dir = '../.' + pre_train_mask_identifier_str_dir
downstream_tasks_pre_train_mask_identifier_constant_rand_dir = '../.' + pre_train_mask_identifier_constant_rand_dir
downstream_tasks_pre_train_mask_identifier_constant_str_dir = '../.' + pre_train_mask_identifier_constant_str_dir
downstream_tasks_pre_train_mask_identifier_rand_str_dir = '../.' + pre_train_mask_identifier_rand_str_dir
downstream_tasks_pre_train_mask_identifier_constant_rand_str_dir = '../.' + pre_train_mask_identifier_constant_rand_str_dir

clone_detection_file_save_dir = downstream_tasks_model_save_dir + 'clone_detection/'
exception_type_file_save_dir = downstream_tasks_model_save_dir + 'exception_type/'
defect_detection_file_save_dir = downstream_tasks_model_save_dir + 'defect_detection/'
code_qa_file_save_dir = downstream_tasks_model_save_dir + 'code_qa/'
code_translation_file_save_dir = downstream_tasks_model_save_dir + 'code_translation/'
code_translation_res_file_dir = code_translation_file_save_dir + 'res_files/'
code_summarization_file_save_dir = downstream_tasks_model_save_dir + 'code_summarization/'
code_summarization_res_file_dir = code_summarization_file_save_dir + 'res_files/'

# -------------------------------other args-------------------------------#

console_sep_string = '-' * 150

model_type = 'roberta'
model_name_or_path = 'microsoft/codebert-base'

# model_type = 'roberta'
# model_name_or_path = 'microsoft/unixcoder-base'

# model_type = 'codet5'
# model_name_or_path = 'Salesforce/codet5-base'

# model_type = 'plbart'
# model_name_or_path = 'uclanlp/plbart-base'
