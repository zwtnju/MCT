import os
import re
from io import StringIO
import tokenize

import os
import sys

sys.path.append(os.path.dirname(sys.path[0]))
from pre_train.utils import args


def root_dir():
    return args.dataset_dir


# for clean old data
def lang_dir_to_generate_useful_features():
    return [args.go_dir, args.java_dir, args.javascript_dir, args.php_dir, args.python_dir, args.ruby_dir,
            args.go_err_dir, args.java_err_dir, args.javascript_err_dir, args.php_err_dir, args.python_err_dir,
            args.ruby_err_dir, ]


def failed_code_dir():
    if not os.path.exists(args.err_dir):
        os.makedirs(args.err_dir)
    return args.err_dir


def data_split_list():
    return ['train', 'valid', 'test']


# for extra features from files
def lang_dir_with_json_files():
    return [args.go_dir, args.java_dir, args.javascript_dir, args.php_dir, args.python_dir, args.ruby_dir]


def lang_dir_to_generate_vocab():
    return [args.go_dir, args.java_dir, args.javascript_dir, args.php_dir, args.python_dir, args.ruby_dir]


def pretrain_lang_dir_list():
    return [args.go_dir, args.java_dir, args.javascript_dir, args.php_dir, args.python_dir, args.ruby_dir]


def downstream_task_data_dir(task=None):
    if task == 'clone detection':
        return args.clone_detection_dir
    elif task == 'exception type':
        return args.exception_type_dir
    elif task == 'defect detection':
        return args.defect_detection_dir
    elif task == 'code qa':
        return args.code_qa_dir
    elif task == 'code translation':
        return args.code_translation_dir
    elif task == 'code summarization':
        return args.code_summarization_dir
    else:
        return None


def pre_trained_model_dir(mask_type):
    if mask_type == 'mask_identifier':
        return args.pre_train_mask_identifier_dir
    if mask_type == 'mask_constant':
        return args.pre_train_mask_constant_dir
    if mask_type == 'mask_var':
        return args.pre_train_mask_var_dir
    if mask_type == 'mask_func_call':
        return args.pre_train_mask_func_call_dir

    if mask_type == 'mask_rand':
        return args.pre_train_mask_rand_dir
    if mask_type == 'mask_str':
        return args.pre_train_mask_str_dir
    if mask_type == 'mask_identifier_constant':
        return args.pre_train_mask_identifier_constant_dir
    if mask_type == 'mask_identifier_rand':
        return args.pre_train_mask_identifier_rand_dir

    if mask_type == 'mask_identifier_str':
        return args.pre_train_mask_identifier_str_dir
    if mask_type == 'mask_identifier_constant_rand':
        return args.pre_train_mask_identifier_constant_rand_dir
    if mask_type == 'mask_identifier_constant_str':
        return args.pre_train_mask_identifier_constant_str_dir
    if mask_type == 'mask_identifier_rand_str':
        return args.pre_train_mask_identifier_rand_str_dir

    if mask_type == 'mask_identifier_constant_rand_str':
        return args.pre_train_mask_identifier_constant_rand_str_dir
    if mask_type == 'mask_constant_var':
        return args.pre_train_mask_constant_var_dir
    if mask_type == 'mask_constant_func':
        return args.pre_train_mask_constant_func_dir
    if mask_type == 'mask_constant_rand':
        return args.pre_train_mask_constant_rand_dir

    if mask_type == 'mask_constant_str':
        return args.pre_train_mask_constant_str_dir
    if mask_type == 'mask_constant_var_func':
        return args.pre_train_mask_constant_var_func_dir
    if mask_type == 'mask_constant_var_rand':
        return args.pre_train_mask_constant_var_rand_dir
    if mask_type == 'mask_constant_var_str':
        return args.pre_train_mask_constant_var_str_dir

    if mask_type == 'mask_constant_func_rand':
        return args.pre_train_mask_constant_func_rand_dir
    if mask_type == 'mask_constant_func_str':
        return args.pre_train_mask_constant_func_str_dir
    if mask_type == 'mask_constant_rand_str':
        return args.pre_train_mask_constant_rand_str_dir
    if mask_type == 'mask_constant_var_func_rand':
        return args.pre_train_mask_constant_var_func_rand_dir

    if mask_type == 'mask_constant_var_func_str':
        return args.pre_train_mask_constant_var_func_str_dir
    if mask_type == 'mask_constant_var_rand_str':
        return args.pre_train_mask_constant_var_rand_str_dir
    if mask_type == 'mask_constant_func_rand_str':
        return args.pre_train_mask_constant_func_rand_str_dir
    if mask_type == 'mask_constant_var_func_rand_str':
        return args.pre_train_mask_constant_var_func_rand_str_dir

    if mask_type == 'mask_var_func':
        return args.pre_train_mask_var_func_dir
    if mask_type == 'mask_var_rand':
        return args.pre_train_mask_var_rand_dir
    if mask_type == 'mask_var_str':
        return args.pre_train_mask_var_str_dir
    if mask_type == 'mask_var_func_rand':
        return args.pre_train_mask_var_func_rand_dir

    if mask_type == 'mask_var_func_str':
        return args.pre_train_mask_var_func_str_dir
    if mask_type == 'mask_var_rand_str':
        return args.pre_train_mask_var_rand_str_dir
    if mask_type == 'mask_var_func_rand_str':
        return args.pre_train_mask_var_func_rand_str_dir
    if mask_type == 'mask_func_var':
        return args.pre_train_mask_func_var_dir

    if mask_type == 'mask_func_rand':
        return args.pre_train_mask_func_rand_dir
    if mask_type == 'mask_func_str':
        return args.pre_train_mask_func_str_dir
    if mask_type == 'mask_func_rand_str':
        return args.pre_train_mask_func_rand_str_dir
    if mask_type == 'mask_rand_str':
        return args.pre_train_mask_rand_str_dir
    return None


# other utils related to data process

def file2list(filename):
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f]
    except IOError:
        return None


def batch_file2list(filenames):
    return list(map(file2list, filenames))


def list2file(filename, data_list, index_range=None):
    try:
        with open(filename, 'a', encoding='utf-8') as f:
            if not index_range:
                for data in data_list:
                    f.write(data + '\n')
            else:
                for i in index_range:
                    f.write(data_list[i] + '\n')
        return True
    except IOError:
        return False


# transform source code to a string line
def code_repr(code):
    return repr(code)[1:-1]


def batch_code_repr(code_list):
    return list(map(code_repr, code_list))


# remove the special /
def code_inverse_repr(code):
    try:
        return code.encode('utf8').decode('unicode_escape')
    except UnicodeDecodeError:
        return code


def batch_code_inverse_repr(code_list):
    return list(map(code_inverse_repr, code_list))


def remove_comments_and_docstrings(source, lang):
    """
    Remove docs and comments from source string.
    Thanks to authors of GraphCodeBERT

    from: https://github.com/microsoft/CodeBERT/blob/master/GraphCodeBERT/codesearch/parser/utils.py#L4

    Args:
        source (str): Source code string
        lang (str): Source code language

    Returns:
        str: Source string

    """
    if lang == args.python:
        try:
            io_obj = StringIO(source)
            out = ""
            prev_token_type = tokenize.INDENT
            last_lineno = -1
            last_col = 0
            for tok in tokenize.generate_tokens(io_obj.readline):
                token_type = tok[0]
                token_string = tok[1]
                start_line, start_col = tok[2]
                end_line, end_col = tok[3]
                # l_text = tok[4]
                if start_line > last_lineno:
                    last_col = 0
                if start_col > last_col:
                    out += (" " * (start_col - last_col))
                # Remove comments:
                if token_type == tokenize.COMMENT:
                    pass
                # This series of conditionals removes docstrings:
                elif token_type == tokenize.STRING:
                    if prev_token_type != tokenize.INDENT:
                        # This is likely a docstring; double-check we're not inside an operator:
                        if prev_token_type != tokenize.NEWLINE:
                            if start_col > 0:
                                out += token_string
                else:
                    out += token_string
                prev_token_type = token_type
                last_col = end_col
                last_lineno = end_line
            temp = []
            for x in out.split('\n'):
                if x.strip() != "":
                    temp.append(x)
            return '\n'.join(temp)
        except Exception:
            return source
    elif lang == args.ruby:

        return source
    else:
        def replacer(match):
            s = match.group(0)
            if s.startswith('/'):
                return " "  # note: a space and not an empty string
            else:
                return s

        pattern = re.compile(
            r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
            re.DOTALL | re.MULTILINE
        )
        temp = []
        for x in re.sub(pattern, replacer, source).split('\n'):
            if x.strip() != "":
                temp.append(x)
        return '\n'.join(temp)


def replace_string_literal(source):
    """
    Replace the string literal in source code with ``<STR>``.

    Args:
        source (str): Source code in string

    Returns:
        str: Code after replaced

    copy from https://github.com/NougatCA/SPT-Code/blob/main/sources/data/data_utils.py
    """

    return re.sub(pattern=re.compile(r'([bruf]*)(\"\"\"|\'\'\'|\"|\')(?:(?!\2)(?:\\.|[^\\]))*\2'),
                  repl=args.string_replaced_token,
                  string=source)


def clean_doc(s):
    """
    Clean docstring.

    Args:
        s (str): Raw docstring

    Returns:
        str: Cleaned docstring

    copy from https://github.com/NougatCA/SPT-Code/blob/main/sources/data/data_utils.py#L522

    """
    # // Create an instance of  {@link RepresentationBaseType } and {@link RepresentationBaseType }.
    # // Create an instance of RepresentationBaseType and RepresentationBaseType
    # // Public setter for the  {@code rowMapper}.
    # // Public setter for the rowMapper
    # comment = comment.replaceAll("\\{@link|code(.*?)}", "$1");
    # comment = comment.replaceAll("@see", "");

    s = re.sub(r'{@link|code(.*?)}', r'\1', s)
    s = re.sub(r'@see', '', s)

    # // Implementation of the <a href="http://www.tarsnap.com/scrypt/scrypt.pdf"/>scrypt KDF</a>.
    # // Implementation of the scrypt KDF
    # comment = comment.replaceAll("<a.*?>(.*?)a>", "$1");
    s = re.sub(r'<a.*?>(.*?)a>', r'\1', s)

    # // remove all tags like <p>, </b>
    # comment = comment.replaceAll("</?[A-Za-z0-9]+>", "");
    s = re.sub(r'</?[A-Za-z0-9]+>', '', s)

    # // Set the list of the watchable objects (meta data).
    # // Set the list of the watchable objects
    # comment = comment.replaceAll("\\(.*?\\)", "");
    s = re.sub(r'\(.*?\)', '', s)

    # // #dispatchMessage dispatchMessage
    # // dispatchMessage
    # comment = comment.replaceAll("#([\\w]+)\\s+\\1", "$1");
    s = re.sub(r'#([\w]+)\s+\1', r'\1', s)

    # // remove http url
    # comment = comment.replaceAll("http\\S*", "");
    s = re.sub(r'http\S*', '', s)

    # // characters except english and number are ignored.
    # comment = comment.replaceAll("[^a-zA-Z0-9_]", " ");
    s = re.sub(r'[^a-zA-Z0-9_]', ' ', s)

    # // delete empty symbols
    # comment = comment.replaceAll("[ \f\n\r\t]", " ").trim();
    # comment = comment.replaceAll(" +", " ");
    s = re.sub(r'[ \f\n\r\t]', ' ', s).strip()
    s = re.sub(r' +', ' ', s).strip()

    if len(s) == 0 or len(s.split()) < 3:
        return args.doc_empty_token
    else:
        return s


def split_vars_with_camel(identifier):
    if identifier == '\n' or identifier == '\t':
        return identifier

    matches = re.findall('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', identifier)
    # if len(matches) > 1:
    #     return [matches[0]] + ['##' + m for m in matches[1:]]
    # else:
    #     return matches
    return matches


def split_vars_with_underline(identifier):
    ret_words = re.sub(r'([a-zA-Z0-9]+)', r' \1 ', identifier).strip().split(" ")

    return ret_words


def split_doc(doc, max_length):
    doc = re.sub(r'([a-zA-Z0-9_*]+)', r' \1 ', doc)
    doc = re.sub(r'(\W)', r' \1 ', doc)
    doc = re.sub(r' +', r' ', doc)

    split_docs = doc.strip().split(" ")
    split_words = []
    ret_docs = []
    for identifier in split_docs:
        if identifier:
            underline_vars = split_vars_with_underline(identifier)
        else:
            underline_vars = []
        # print(underline_vars)
        for unl_var in underline_vars:
            camel_vars = split_vars_with_camel(unl_var)
            # print(camel_vars)
            split_words.extend(camel_vars)

    i = 0
    while i < len(split_words) - 1:
        if split_words[i] == split_words[i + 1] and split_words[i] in args.special_double_symbol_list:
            ret_docs.append(split_words[i] + split_words[i + 1])
            i += 2
        else:
            ret_docs.append(split_words[i])
            i += 1

    ret_docs.append(split_words[-1])
    doc_len = len(ret_docs)
    if not max_length == -1:
        ret_docs = ret_docs[: min(doc_len, max_length)]
    return ret_docs


def batch_split_doc(docs, max_length=args.max_nl_len):
    return list(map(lambda x: split_doc(x, max_length), docs))


def split_code(code):
    """
    Split code into a list of subtokens.

    Args:
        code (str): given code

    Returns:
        list[str]: list of subtokens
    """
    #
    code = re.sub(r'([a-zA-Z0-9_*]+)', r' \1 ', code)
    code = re.sub(r'(\W)', r' \1 ', code)
    code = re.sub(r' +', r' ', code)

    split_codes = code.strip().split(" ")
    split_words = []
    for identifier in split_codes:
        underline_vars = split_vars_with_underline(identifier)
        for unl_var in underline_vars:
            camel_vars = split_vars_with_camel(unl_var)
            split_words.extend(camel_vars)

    ret_words = []
    i = 0

    while i < len(split_words) - 1:
        if split_words[i] == split_words[i + 1] and split_words[i] in args.special_double_symbol_list:
            ret_words.append(split_words[i] + split_words[i + 1])
            i += 2

        # deal with special tokens e.g.
        # '[', 'STR', ']'
        elif split_words[i] == '[' and split_words[i + 1] == 'TZLG':
            assert split_words[i + 2] == ']', f'var replace tokenizer fail'
            var_token = split_words[i] + split_words[i + 1] + split_words[i + 2]
            ret_words.append(var_token)
            i += 3

        # there are duplicate data start with [STR]
        elif split_words[i] == '[' and split_words[i + 1] == 'NCAST':
            assert split_words[i + 2] == ']', f'str replace tokenizer fail'
            str_token = split_words[i] + split_words[i + 1] + split_words[i + 2]
            ret_words.append(str_token)
            i += 3
        else:
            ret_words.append(split_words[i])
            i += 1

    ret_words.append(split_words[-1])
    return ret_words


def batch_split_code(codes):
    return list(map(split_code, codes))
