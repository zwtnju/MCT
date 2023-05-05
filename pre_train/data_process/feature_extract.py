import json
import os
import re

import pre_train.utils.data as data_utils
from ast_parser import extract_variable_from_code, extract_function_call_from_code, \
    extract_identifier_from_code, extract_string_from_code
from pre_train.utils import args


# extra the features we need for results inputs
def extract_useful_code_features(dataset_lang_dir, fail_dir, lang, is_train='train'):
    lang_dir = f"{dataset_lang_dir}{is_train}"
    lang_err_dir = f"{fail_dir}{lang}/{is_train}"

    print(f'correct {is_train} code lines are stored in: {lang_dir}')
    print(f'error {is_train} code lines are stored in: {lang_err_dir}')

    for file_dir in [lang_dir, lang_err_dir]:
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)

    raw_codes = data_utils.file2list(f"{lang_dir}/code")
    urls = data_utils.file2list(f"{lang_dir}/url")
    docs = data_utils.file2list(f"{lang_dir}/doc")
    assert len(raw_codes) == len(urls) == len(docs), f"extract features from files error"

    valid_code_file = f'{lang_dir}/{args.valid_code_file_name}'
    valid_doc_file = f'{lang_dir}/{args.valid_doc_file_name}'

    mask_var_str_code_file = f'{lang_dir}/{args.mask_var_str_code_file_name}'
    code_vars_file = f'{lang_dir}/{args.code_variables_file_name}'
    mask_func_call_str_code_file = f'{lang_dir}/{args.mask_function_call_str_code_file_name}'
    code_func_calls_file = f'{lang_dir}/{args.code_function_calls_file_name}'
    mask_identifier_str_code_file = f'{lang_dir}/{args.mask_identifier_str_code_file_name}'
    code_identifier_file = f'{lang_dir}/{args.codes_identifiers_file_name}'

    mask_str_code_file = f'{lang_dir}/{args.mask_str_code_file_name}'
    code_strings_file = f'{lang_dir}/{args.code_strings_file_name}'
    err_code_file = f"{lang_err_dir}/{args.err_useful_code_file_name}"
    err_url_file = f"{lang_err_dir}/{args.err_useful_url_file_name}"

    for file in [err_code_file, err_url_file]:
        if os.path.exists(file):
            os.remove(file)

    valid_index = []
    invalid_index = []
    mask_var_str_codes = []
    codes_variables = []
    mask_function_call_str_codes = []
    codes_function_calls = []
    mask_identifier_str_codes = []
    codes_identifiers = []
    mask_str_codes = []
    codes_strings = []

    print('code that cannot be parsed by an abstract syntax tree:')

    for i in range(len(raw_codes)):
        line = raw_codes[i]
        code = data_utils.code_inverse_repr(line)
        try:

            # extract different parts of codes
            mask_var_str_code, mask_str_code1, code_variable, code_strings1 = extract_variable_from_code(code, lang)
            mask_function_call_str_code, mask_str_code2, code_function_call, code_strings2 = extract_function_call_from_code(
                code, lang)
            mask_identifier_str_code, mask_str_code3, code_identifier, code_strings3 = extract_identifier_from_code(
                code, lang)
            mask_str_code, code_string = extract_string_from_code(code, lang)

            assert len(code_identifier) == len(code_variable) + len(code_function_call), f"extract data fail"
            assert code_strings1 == code_strings2 == code_strings3 == code_string, f"extract data fail"

            mask_var_str_codes.append(mask_var_str_code)
            codes_variables.append(code_variable)
            mask_function_call_str_codes.append(mask_function_call_str_code)
            codes_function_calls.append(code_function_call)
            mask_identifier_str_codes.append(mask_identifier_str_code)
            codes_identifiers.append(code_identifier)
            mask_str_codes.append(mask_str_code)
            codes_strings.append(code_string[1:-1])
            valid_index.append(i)
        except AssertionError:
            invalid_index.append(i)
            print(f"error occurs on {i} code of {lang} code {args.console_sep_string}\n {code}")

    assert len(mask_var_str_codes) == len(codes_variables) == len(mask_function_call_str_codes) == \
           len(codes_function_calls) == len(mask_identifier_str_codes) == len(codes_identifiers) == \
           len(mask_str_codes) == len(codes_strings) == len(valid_index), f"parse ast fail"

    # ast extract success
    assert data_utils.list2file(valid_code_file, raw_codes, valid_index), f"write file error"
    print(f"append to {valid_code_file}")

    assert data_utils.list2file(valid_doc_file, docs, valid_index), f"write file error"
    print(f"append to {valid_doc_file}")

    assert data_utils.list2file(mask_var_str_code_file,
                                data_utils.batch_code_repr(mask_var_str_codes)), f"write file error"
    print(f"append to {mask_var_str_code_file}")

    assert data_utils.list2file(mask_func_call_str_code_file,
                                data_utils.batch_code_repr(mask_function_call_str_codes)), f"write file error"
    print(f"append to {mask_func_call_str_code_file}")

    assert data_utils.list2file(mask_identifier_str_code_file,
                                data_utils.batch_code_repr(mask_identifier_str_codes)), f"write file error"
    print(f"append to {mask_identifier_str_code_file}")

    assert data_utils.list2file(mask_str_code_file,
                                data_utils.batch_code_repr(mask_str_codes)), f"write file error"
    print(f"append to {mask_str_code_file}")

    print(f"append to {code_vars_file}")
    with open(code_vars_file, 'a', encoding='utf-8') as f_write_code_vars:
        for i in range(len(codes_variables)):
            code_vars = repr(','.join(codes_variables[i]))[1:-1] + '\n'
            f_write_code_vars.write(code_vars)

    print(f"append to {code_func_calls_file}")
    with open(code_func_calls_file, 'a', encoding='utf-8') as f_write_code_vars:
        for i in range(len(codes_function_calls)):
            code_func_call = repr(','.join(codes_function_calls[i]))[1:-1] + '\n'
            f_write_code_vars.write(code_func_call)

    print(f"append to {code_identifier_file}")
    with open(code_identifier_file, 'a', encoding='utf-8') as f_write_code_vars:
        for i in range(len(codes_identifiers)):
            code_identifier = repr(','.join(codes_identifiers[i]))[1:-1] + '\n'
            f_write_code_vars.write(code_identifier)

    print(f"append to {code_strings_file}")
    with open(code_strings_file, 'a', encoding='utf-8') as f_write_code_strings:
        for i in range(len(codes_strings)):
            code_str = repr(' '.join(codes_strings[i]))[1:-1] + '\n'
            f_write_code_strings.write(code_str)

    # ast extract error
    if invalid_index:
        assert data_utils.list2file(err_code_file, raw_codes, invalid_index), f"write file error"
        print(f"record {lang} err code in {err_code_file}")

        assert data_utils.list2file(err_url_file, urls, invalid_index), f"write file error"
        print(f"record {lang} err url in {err_url_file}")

        with open(err_code_file, 'a', encoding='utf-8') as f_write_num:
            f_write_num.write(str(len(raw_codes)))
    else:
        print(f"no err codes when parsed to ast")


# This function is to extra features from dataset to the mode that we need
def batch_extract_useful_code_features(lang_dirs, fail_dir, data_split):
    print(args.console_sep_string)
    print('start to extract useful code features from raw code files')
    for lang_dir in lang_dirs:
        for is_train in data_split:
            lang = lang_dir.split('/')[-2]
            assert lang in args.langs
            print(f'process {lang} {is_train} dataset start')
            extract_useful_code_features(lang_dir, fail_dir, lang, is_train)
            print(args.console_sep_string)

    print("extract useful code features finished")
    print(args.console_sep_string)


def extract_code_features_from_file(lang_dir, file_name, fail_dir, lang, is_train='train'):
    lang_err_dir = f"{fail_dir}{lang}/{is_train}"

    print(f'correct {is_train} code lines are stored in: {lang_dir}')
    print(f'error {is_train} code lines are stored in: {lang_err_dir}')

    for file_dir in [lang_err_dir]:
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)

    code_file = f"{lang_dir}/{args.code_file_name}"
    doc_file = f"{lang_dir}/{args.doc_file_name}"
    url_file = f"{lang_dir}/{args.url_file_name}"
    err_code_file = f"{lang_err_dir}/{args.err_code_file_name}"
    err_url_file = f"{lang_err_dir}/{args.err_url_file_name}"

    for file in [err_code_file, err_url_file]:
        if os.path.exists(file):
            os.remove(file)

    with open(os.path.join(lang_dir, file_name), 'r', encoding='utf-8') as raw_file:
        code_fields = list(map(lambda x: json.loads(x), raw_file))

    write_codes = []
    write_docs = []
    write_urls = []
    write_fail_codes = []
    write_fail_urls = []

    for code_field in code_fields:
        code = code_field[args.code]
        doc = code_field[args.doc]
        url = code_field[args.url]

        special_pattern = r"\\x[a-zA-Z0-9]*"
        match_str = repr(code.encode('utf-8')) + repr(doc.encode('utf-8'))
        code_pattern = re.findall(special_pattern, match_str)
        if not code_pattern:

            # clean code and doc
            # code = data_utils.remove_comments_and_docstrings(code, lang)
            # code = data_utils.replace_string_literal(code)
            # doc = data_utils.clean_doc(doc)

            code = data_utils.code_repr(code)
            doc = data_utils.code_repr(doc)
            url = data_utils.code_repr(url)

            write_codes.append(code)
            write_docs.append(doc)
            write_urls.append(url)
        else:
            code = data_utils.code_repr(code)
            url = data_utils.code_repr(url)

            write_fail_codes.append(code)
            write_fail_urls.append(url)

    assert data_utils.list2file(code_file, write_codes), f"write file error"
    print(f"append 'code' column to {code_file}")

    assert data_utils.list2file(doc_file, write_docs), f"write file error"
    print(f"append 'doc' column to {doc_file}")

    assert data_utils.list2file(url_file, write_urls), f"write file error"
    print(f"append 'url' column to {url_file}")

    assert data_utils.list2file(err_code_file, write_fail_codes), f"write file error"
    print(f"append error 'code' column to {err_code_file}")

    assert data_utils.list2file(err_url_file, write_fail_urls), f"write file error"
    print(f"append error 'url' column to {err_url_file}")

    with open(err_code_file, 'a', encoding='utf-8') as f_write_num:
        f_write_num.write(str(len(write_codes) + len(write_fail_codes)))


def batch_extract_code_features_from_file(lang_dirs, fail_dir):

    print(args.console_sep_string)
    # Then generate data
    print('start to extract code features from json files')
    for lang_dir in lang_dirs:
        print(args.console_sep_string)
        print(f'current language: {lang_dir}')
        for is_train in os.listdir(lang_dir):
            data_split_dir = os.path.join(lang_dir, is_train)
            print(f'current directory/file: {data_split_dir}')
            if os.path.isdir(data_split_dir):
                for file_name in os.listdir(data_split_dir):
                    lang = file_name.split('_')[0]
                    assert lang in args.langs
                    if file_name.endswith('.jsonl'):
                        print(f'extract features from {file_name}')
                        extract_code_features_from_file(data_split_dir, file_name, fail_dir, lang, is_train)
                        print(args.console_sep_string)

    print('generate code features finished')
    print(args.console_sep_string)
