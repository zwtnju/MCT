import pickle
import random
from transformers.utils import logging

from torch.utils.data import Dataset, Subset

import utils.data as data_utils
from utils import args

logger = logging.get_logger(__name__)


class CodeMAEDataset(Dataset):

    def __init__(self, data_dir, is_train='train'):

        valid_codes_dir = []
        valid_docs_dir = []

        mask_var_str_codes_dir = []
        codes_vars_dir = []

        mask_function_call_str_codes_dir = []
        codes_function_calls_dir = []

        mask_identifier_str_codes_dir = []
        codes_identifiers_dir = []

        mask_str_codes_dir = []
        codes_strings_dir = []

        if is_train == 'all':
            for data_split in data_utils.data_split_list():
                valid_codes_dir = valid_codes_dir + [f"{lang_dir}{data_split}/{args.valid_code_file_name}" for lang_dir
                                                     in data_dir]
                valid_docs_dir = valid_docs_dir + [f"{lang_dir}{data_split}/{args.valid_doc_file_name}" for lang_dir in
                                                   data_dir]

                mask_var_str_codes_dir = mask_var_str_codes_dir + [
                    f"{lang_dir}{data_split}/{args.mask_var_str_code_file_name}" for lang_dir in data_dir]
                codes_vars_dir = codes_vars_dir + [f"{lang_dir}{data_split}/{args.code_variables_file_name}" for
                                                   lang_dir in data_dir]

                mask_function_call_str_codes_dir = mask_function_call_str_codes_dir + [
                    f"{lang_dir}{data_split}/{args.mask_function_call_str_code_file_name}" for lang_dir in data_dir]
                codes_function_calls_dir = codes_function_calls_dir + [
                    f"{lang_dir}{data_split}/{args.code_function_calls_file_name}" for lang_dir in data_dir]

                mask_identifier_str_codes_dir = mask_identifier_str_codes_dir + [
                    f"{lang_dir}{data_split}/{args.mask_identifier_str_code_file_name}" for lang_dir in data_dir]
                codes_identifiers_dir = codes_identifiers_dir + [
                    f"{lang_dir}{data_split}/{args.codes_identifiers_file_name}" for lang_dir in data_dir]

                mask_str_codes_dir = mask_str_codes_dir + [f"{lang_dir}{data_split}/{args.mask_str_code_file_name}" for
                                                           lang_dir in data_dir]
                codes_strings_dir = codes_strings_dir + [f"{lang_dir}{data_split}/{args.code_strings_file_name}" for
                                                         lang_dir in data_dir]
        else:
            assert is_train in data_utils.data_split_list()
            valid_codes_dir = [f"{lang_dir}{is_train}/{args.valid_code_file_name}" for lang_dir in data_dir]
            valid_docs_dir = [f"{lang_dir}{is_train}/{args.valid_doc_file_name}" for lang_dir in data_dir]

            mask_var_str_codes_dir = [f"{lang_dir}{is_train}/{args.mask_var_str_code_file_name}" for lang_dir in
                                      data_dir]
            codes_vars_dir = [f"{lang_dir}{is_train}/{args.code_variables_file_name}" for lang_dir in data_dir]

            mask_function_call_str_codes_dir = [f"{lang_dir}{is_train}/{args.mask_function_call_str_code_file_name}" for
                                                lang_dir in data_dir]
            codes_function_calls_dir = [f"{lang_dir}{is_train}/{args.code_function_calls_file_name}" for lang_dir in
                                        data_dir]

            mask_identifier_str_codes_dir = [f"{lang_dir}{is_train}/{args.mask_identifier_str_code_file_name}" for
                                             lang_dir in data_dir]
            codes_identifiers_dir = [f"{lang_dir}{is_train}/{args.codes_identifiers_file_name}" for lang_dir in
                                     data_dir]

            mask_str_codes_dir = [f"{lang_dir}{is_train}/{args.mask_str_code_file_name}" for lang_dir in data_dir]
            codes_strings_dir = [f"{lang_dir}{is_train}/{args.code_strings_file_name}" for lang_dir in data_dir]

        self.valid_codes = []
        self.valid_docs = []

        self.codes_mask_var = []
        self.codes_variables = []

        self.codes_mask_func_call = []
        self.codes_func_calls = []

        self.codes_mask_identifier = []
        self.codes_identifiers = []

        self.codes_mask_str = []
        self.codes_strings = []

        self.langs = []

        self.is_train = is_train

        for valid_code_file, valid_doc_file, mask_var_str_code_file, mask_function_call_str_code_file, \
            mask_identifier_str_code_file, mask_str_code_file, code_vars_file, code_function_call_file, \
            code_identifier_file, code_strings_file in zip(
                valid_codes_dir, valid_docs_dir, mask_var_str_codes_dir, mask_function_call_str_codes_dir,
                mask_identifier_str_codes_dir, mask_str_codes_dir, codes_vars_dir, codes_function_calls_dir,
                codes_identifiers_dir, codes_strings_dir):
            # encoder input
            valid_docs = data_utils.file2list(valid_doc_file)
            valid_docs = data_utils.batch_code_inverse_repr(valid_docs)
            self.valid_docs.extend(valid_docs)

            mask_var_str_codes = data_utils.file2list(mask_var_str_code_file)
            mask_var_str_codes = data_utils.batch_code_inverse_repr(mask_var_str_codes)
            self.codes_mask_var.extend(mask_var_str_codes)

            mask_function_call_str_codes = data_utils.file2list(mask_function_call_str_code_file)
            mask_function_call_str_codes = data_utils.batch_code_inverse_repr(mask_function_call_str_codes)
            self.codes_mask_func_call.extend(mask_function_call_str_codes)

            mask_identifier_str_codes = data_utils.file2list(mask_identifier_str_code_file)
            mask_identifier_str_codes = data_utils.batch_code_inverse_repr(mask_identifier_str_codes)
            self.codes_mask_identifier.extend(mask_identifier_str_codes)

            # decoder output
            mask_str_codes = data_utils.file2list(mask_str_code_file)
            mask_str_codes = data_utils.batch_code_inverse_repr(mask_str_codes)
            self.codes_mask_str.extend(mask_str_codes)

            # masked parts
            codes_vars = data_utils.file2list(code_vars_file)
            codes_vars = data_utils.batch_code_inverse_repr(codes_vars)
            self.codes_variables.extend(codes_vars)

            codes_function_calls = data_utils.file2list(code_function_call_file)
            codes_function_calls = data_utils.batch_code_inverse_repr(codes_function_calls)
            self.codes_func_calls.extend(codes_function_calls)

            codes_identifiers = data_utils.file2list(code_identifier_file)
            codes_identifiers = data_utils.batch_code_inverse_repr(codes_identifiers)
            self.codes_identifiers.extend(codes_identifiers)

            codes_strings = data_utils.file2list(code_strings_file)
            codes_strings = data_utils.batch_code_inverse_repr(codes_strings)
            self.codes_strings.extend(codes_strings)

            # other useless items
            valid_codes = data_utils.file2list(valid_code_file)
            code_nums = len(valid_codes)
            valid_codes = data_utils.batch_code_inverse_repr(valid_codes)
            self.valid_codes.extend(valid_codes)

            lang = valid_doc_file.split('/')[-3]
            langs = [lang] * code_nums
            self.langs.extend(langs)

        assert len(self.valid_docs) == len(self.codes_mask_var) == len(self.codes_mask_str) \
               == len(self.codes_variables) == len(self.codes_strings) == len(self.valid_codes) == len(self.langs)
        self.size = len(self.valid_codes)

    def __getitem__(self, idx):

        """
        :param idx: the item index
        :return: self.valid_docs[idx]: docstrings of code for encoder input
                 self.langs[idx]: code language
                 self.codes_mask_str[idx]: code with strings replaced by a single token for decoder output
                 self.codes_strings[idx]: string lists of a code fragment, used as nl information

                 self.codes_mask_var[idx]: code with variables masked by a single [MASK] token for decoder input
                 self.codes_mask_func_call[idx]: code with function calls masked by a single [MASK] token for decoder input
                 self.codes_mask_identifier[idx]: code with identifiers masked by a single [MASK] token for decoder input

                 self.codes_variables[idx]: variable lists of a code fragment
                 self.codes_func_calls[idx]: function call lists of a code fragment
                 self.codes_identifiers[idx]: identifier lists of a code fragment


        """

        return self.valid_docs[idx], self.langs[idx], self.codes_mask_str[idx], self.codes_strings[idx], \
               self.codes_mask_var[idx], self.codes_mask_func_call[idx], self.codes_mask_identifier[idx], \
               self.codes_variables[idx], self.codes_func_calls[idx], self.codes_identifiers[idx],

    def __len__(self):
        return self.size

    def subset(self, ratio):
        indices = random.sample(range(self.size), int(self.size * ratio))
        return Subset(self, indices)


def generate_original_dataset():
    data_dir = data_utils.pretrain_lang_dir_list()
    train_data = CodeMAEDataset(['.' + x for x in data_dir], is_train='all')
    train_file = open("train_data.pkl", "wb")
    pickle.dump(train_data, train_file)
    train_file.close()

# generate_original_dataset()
