import logging
import random
import re

import torch

from utils import data as data_utils


def mask_random(doc, lang, code_mask_str, code_strings, tokenizer, args):
    if doc == '[NONE]':
        doc = ''
    doc = lang + ' ' + doc + ' ' + code_strings
    doc = ' '.join(data_utils.split_doc(doc, args.max_nl_len))
    code_mask_str = ' '.join(data_utils.split_code(code_mask_str)).replace('[NCAST]', '').replace('[TZLG]', '')
    code_mask_str = re.sub(r' +', r' ', code_mask_str)
    vocab_size = tokenizer.vocab_size

    doc_tokens = tokenizer.tokenize(doc)[:args.max_nl_len]
    if args.model_type == 'codebert' or args.model_type == 'unixcoder':
        str_tokens = ([tokenizer.cls_token] + doc_tokens + [tokenizer.sep_token] +
                      tokenizer.tokenize(code_mask_str) + [tokenizer.eos_token])[:args.max_seq_len]
        token_type_ids = ([0] * (len(doc_tokens) + 2) + [1] * args.max_seq_len)[:args.max_seq_len]
    else:
        str_tokens = (doc_tokens + [tokenizer.sep_token] + tokenizer.tokenize(code_mask_str))[:args.max_seq_len]
        token_type_ids = ([0] * (len(doc_tokens) + 1) + [1] * args.max_seq_len)[:args.max_seq_len]

    str_ids = tokenizer.convert_tokens_to_ids(str_tokens)
    str_mask = [1] * (len(str_tokens))
    padding_length = args.max_seq_len - len(str_ids)
    str_ids += [tokenizer.pad_token_id] * padding_length
    str_mask += [0] * padding_length

    input_ids = str_ids
    attention_mask = str_mask

    if len(str_tokens) == args.max_seq_len:
        label_tokens = str_tokens[: len(str_tokens) - 1] + [tokenizer.eos_token]
    else:
        label_tokens = str_tokens + [tokenizer.eos_token]
    label = tokenizer.convert_tokens_to_ids(label_tokens)
    padding_length = args.max_seq_len - len(label_tokens)
    label += [args.ignore_index] * padding_length

    # keep the same with bert
    for c in range(len(str_tokens)):
        rand = random.random()
        if rand < 0.15:
            rand /= 0.15
            if rand > 0.2:
                attention_mask[c] = 0
                input_ids[c] = tokenizer.mask_token_id
            elif rand < 0.1:
                random_id = random.randint(0, vocab_size - 1)
                input_ids[c] = random_id
            else:
                input_ids[c] = input_ids[c]
        else:
            label[c] = args.ignore_index
    assert len(token_type_ids) == len(input_ids) == len(attention_mask) == len(
        label) == args.max_seq_len, f"embedding error"
    source_ids = input_ids
    source_token_type_id = token_type_ids
    source_mask = attention_mask
    target_ids = label

    tgt_ids = []
    for i in label:
        if i >= 0:
            tgt_ids.append(i)

    target_token = tokenizer.convert_ids_to_tokens(tgt_ids)
    if len(target_token) == 0:
        return None, None, None, None
    return source_ids, source_token_type_id, source_mask, target_ids


def mask_constant(doc, lang, code_mask_str, code_strings, code_mask_identifier, code_identifiers,
                  tokenizer, args):
    if args.model_type == 'codebert' or args.model_type == 'unixcoder' or args.model_type == 'codet5':
        special_split_symbol = 'Ġ'
    elif args.model_type == 'plbart':
        special_split_symbol = '▁'
    else:
        special_split_symbol = None
    if not special_split_symbol:
        return None, None, None, None
    if doc == '[NONE]':
        doc = ''
    doc = lang + ' ' + doc + ' ' + code_strings

    doc = ' '.join(data_utils.split_doc(doc, args.max_nl_len))
    code_mask_var = ' '.join(data_utils.split_code(code_mask_identifier)).replace('[NCAST]', '').replace('[TZLG]', '')
    code_mask_var = re.sub(r' +', r' ', code_mask_var)
    code_mask_str = ' '.join(data_utils.split_code(code_mask_str)).replace('[NCAST]', '').replace('[TZLG]', '')
    code_mask_str = re.sub(r' +', r' ', code_mask_str)
    var_list = code_identifiers.split(',')
    vocab_size = tokenizer.vocab_size

    doc_tokens = tokenizer.tokenize(doc)[:args.max_nl_len]

    if args.model_type == 'codebert' or args.model_type == 'unixcoder':
        var_tokens = ([tokenizer.cls_token] + doc_tokens + [tokenizer.sep_token] +
                      tokenizer.tokenize(code_mask_var))[:args.max_seq_len - 1] + [tokenizer.eos_token]
        str_tokens = ([tokenizer.cls_token] + doc_tokens + [tokenizer.sep_token] +
                      tokenizer.tokenize(code_mask_str))[:args.max_seq_len - 1] + [tokenizer.eos_token]

        token_type_ids = ([0] * (len(doc_tokens) + 2) + [1] * args.max_seq_len)[:args.max_seq_len]
    else:
        var_tokens = (doc_tokens + [tokenizer.sep_token] + tokenizer.tokenize(code_mask_var))[:args.max_seq_len]
        str_tokens = (doc_tokens + [tokenizer.sep_token] + tokenizer.tokenize(code_mask_str))[:args.max_seq_len]

        token_type_ids = ([0] * (len(doc_tokens) + 1) + [1] * args.max_seq_len)[:args.max_seq_len]

    var_ids = tokenizer.convert_tokens_to_ids(var_tokens)
    var_mask = [1] * (len(var_tokens))
    padding_length = args.max_seq_len - len(var_ids)
    var_ids += [tokenizer.pad_token_id] * padding_length
    var_mask += [0] * padding_length

    str_ids = tokenizer.convert_tokens_to_ids(str_tokens)
    str_mask = [1] * (len(str_tokens))
    padding_length = args.max_seq_len - len(str_ids)
    str_ids += [tokenizer.pad_token_id] * padding_length
    str_mask += [0] * padding_length

    input_ids = str_ids
    attention_mask = str_mask

    if args.model_type == 'codebert' or args.model_type == 'unixcoder':
        label_tokens = str_tokens[:]
    else:
        if len(str_tokens) == args.max_seq_len:
            label_tokens = str_tokens[: len(str_tokens) - 1] + [tokenizer.eos_token]
        else:
            label_tokens = str_tokens + [tokenizer.eos_token]
    label = tokenizer.convert_tokens_to_ids(label_tokens)
    padding_length = args.max_seq_len - len(label_tokens)
    label += [args.ignore_index] * padding_length

    if tokenizer.sep_token_id in input_ids:
        sep_index = input_ids.index(tokenizer.sep_token_id)
    else:
        sep_index = 0

    c, m, v = sep_index + 1, sep_index + 1, 0
    for ignore_str_index in range(sep_index + 1):
        label[ignore_str_index] = args.ignore_index

    err_flag = False
    local_var = var_list[v]
    if len(code_identifiers) > 0:
        while c < len(str_tokens):
            if input_ids[c] == tokenizer.pad_token_id:
                break
            if var_tokens[m] == str_tokens[c] or (special_split_symbol + var_tokens[m]) == str_tokens[c]:
                rand = random.random()
                if rand > 0.2:
                    attention_mask[c] = 0
                    input_ids[c] = tokenizer.mask_token_id
                elif rand < 0.1:
                    random_id = random.randint(0, vocab_size - 1)
                    input_ids[c] = random_id
                else:
                    input_ids[c] = input_ids[c]

                m += 1
                c += 1

            elif (special_split_symbol + var_tokens[m]).startswith(str_tokens[c]):
                var_tokens[m] = var_tokens[m][len(str_tokens[c].replace(special_split_symbol, '')):]
                rand = random.random()
                if rand > 0.2:
                    attention_mask[c] = 0
                    input_ids[c] = tokenizer.mask_token_id
                elif rand < 0.1:
                    random_id = random.randint(0, vocab_size - 1)
                    input_ids[c] = random_id
                else:
                    input_ids[c] = input_ids[c]
                c += 1

            elif var_tokens[m].startswith(str_tokens[c]):
                var_tokens[m] = var_tokens[m][len(str_tokens[c]):]
                rand = random.random()
                if rand > 0.2:
                    attention_mask[c] = 0
                    input_ids[c] = tokenizer.mask_token_id
                elif rand < 0.1:
                    random_id = random.randint(0, vocab_size - 1)
                    input_ids[c] = random_id
                else:
                    input_ids[c] = input_ids[c]
                c += 1

            elif str_tokens[c].startswith((special_split_symbol + var_tokens[m])):
                str_tokens[c] = str_tokens[c][len(special_split_symbol + var_tokens[m]):]
                m += 1

            elif str_tokens[c].startswith((var_tokens[m])):
                str_tokens[c] = str_tokens[c][len(var_tokens[m]):]
                m += 1

            else:
                try:
                    local_var = local_var.replace(' ', '')

                    while local_var:
                        assert local_var.startswith(str_tokens[c]) or (special_split_symbol + local_var).startswith(
                            str_tokens[c]), "error"
                        if local_var.startswith(str_tokens[c]):
                            local_var = local_var[len(str_tokens[c]):]
                        else:
                            local_var = local_var[len(str_tokens[c].replace(special_split_symbol, '')):]

                        label[c] = args.ignore_index
                        c += 1
                        if c == len(str_tokens):  # code has been truncated
                            break
                except (AssertionError, IndexError):
                    err_flag = True
                    break

                v += 1
                if v == len(var_list):
                    break
                local_var = var_list[v]

            if m == len(var_tokens) or c == len(str_tokens):
                break
    else:
        # keep the same with bert
        for c in range(len(str_tokens)):
            rand = random.random()
            if rand < 0.15:
                rand /= 0.15

                if rand > 0.2:
                    attention_mask[c] = 0
                    input_ids[c] = tokenizer.mask_token_id
                elif rand < 0.1:
                    random_id = random.randint(0, vocab_size - 1)
                    input_ids[c] = random_id
                else:
                    input_ids[c] = input_ids[c]
            else:
                label[c] = args.ignore_index

    if not err_flag:
        assert len(token_type_ids) == len(input_ids) == len(attention_mask) == len(
            label) == args.max_seq_len, f"embedding error"
        source_ids = input_ids
        source_token_type_id = token_type_ids
        source_mask = attention_mask
        target_ids = label

        tgt_ids = []
        for i in label:
            if i >= 0:
                tgt_ids.append(i)

        target_token = tokenizer.convert_ids_to_tokens(tgt_ids)
        if len(target_token) == 0:
            return None, None, None, None
        return source_ids, source_token_type_id, source_mask, target_ids

    else:
        return None, None, None, None


def mask_string(doc, lang, code_mask_str, code_strings, tokenizer, args):
    if doc == '[NONE]':
        doc = ''
    doc = lang + ' ' + doc + ' ' + code_strings
    doc = ' '.join(data_utils.split_doc(doc, args.max_nl_len))
    code_mask_str = ' '.join(data_utils.split_code(code_mask_str)).replace('[NCAST]', '').replace('[TZLG]', '')
    code_mask_str = re.sub(r' +', r' ', code_mask_str)
    vocab_size = tokenizer.vocab_size

    doc_tokens = tokenizer.tokenize(doc)[:args.max_nl_len]
    if args.model_type == 'codebert' or args.model_type == 'unixcoder':
        str_tokens = ([tokenizer.cls_token] + doc_tokens + [tokenizer.sep_token] +
                      tokenizer.tokenize(code_mask_str))[:args.max_seq_len - 1] + [tokenizer.eos_token]
        token_type_ids = ([0] * (len(doc_tokens) + 2) + [1] * args.max_seq_len)[:args.max_seq_len]
    else:
        str_tokens = (doc_tokens + [tokenizer.sep_token] + tokenizer.tokenize(code_mask_str))[:args.max_seq_len]
        token_type_ids = ([0] * (len(doc_tokens) + 1) + [1] * args.max_seq_len)[:args.max_seq_len]

    str_ids = tokenizer.convert_tokens_to_ids(str_tokens)
    str_mask = [1] * (len(str_tokens))
    padding_length = args.max_seq_len - len(str_ids)
    str_ids += [tokenizer.pad_token_id] * padding_length
    str_mask += [0] * padding_length

    input_ids = str_ids
    attention_mask = str_mask

    if args.model_type == 'codebert' or args.model_type == 'unixcoder':
        label_tokens = str_tokens[:]
    else:
        if len(str_tokens) == args.max_seq_len:
            label_tokens = str_tokens[: len(str_tokens) - 1] + [tokenizer.eos_token]
        else:
            label_tokens = str_tokens + [tokenizer.eos_token]

    label = tokenizer.convert_tokens_to_ids(label_tokens)
    padding_length = args.max_seq_len - len(label_tokens)
    label += [args.ignore_index] * padding_length

    if tokenizer.sep_token_id in input_ids:
        sep_index = input_ids.index(tokenizer.sep_token_id)
    else:
        sep_index = 0

    # keep the same with bert
    for c in range(sep_index):
        rand = random.random()
        if rand < 0.15:
            rand /= 0.15
            if rand > 0.2:
                attention_mask[c] = 0
                input_ids[c] = tokenizer.mask_token_id
            elif rand < 0.1:
                random_id = random.randint(0, vocab_size - 1)
                input_ids[c] = random_id
            else:
                input_ids[c] = input_ids[c]
        else:
            label[c] = args.ignore_index
    assert len(token_type_ids) == len(input_ids) == len(attention_mask) == len(
        label) == args.max_seq_len, f"embedding error"
    source_ids = input_ids
    source_token_type_id = token_type_ids
    source_mask = attention_mask
    target_ids = label

    tgt_ids = []
    for i in label:
        if i >= 0:
            tgt_ids.append(i)

    target_token = tokenizer.convert_ids_to_tokens(tgt_ids)
    if len(target_token) == 0:
        return None, None, None, None
    return source_ids, source_token_type_id, source_mask, target_ids


def mask_nonrandom(doc, lang, code_mask_str, code_strings, code_mask_var, code_variables, tokenizer, args):
    if args.model_type == 'codebert' or args.model_type == 'unixcoder' or args.model_type == 'codet5':
        special_split_symbol = 'Ġ'
    elif args.model_type == 'plbart':
        special_split_symbol = '▁'
    else:
        special_split_symbol = None
    if not special_split_symbol:
        return None, None, None, None
    if doc == '[NONE]':
        doc = ''
    doc = lang + ' ' + doc + ' ' + code_strings

    doc = ' '.join(data_utils.split_doc(doc, args.max_nl_len))
    code_mask_var = ' '.join(data_utils.split_code(code_mask_var)).replace('[NCAST]', '').replace('[TZLG]', '')
    code_mask_var = re.sub(r' +', r' ', code_mask_var)
    code_mask_str = ' '.join(data_utils.split_code(code_mask_str)).replace('[NCAST]', '').replace('[TZLG]', '')
    code_mask_str = re.sub(r' +', r' ', code_mask_str)
    var_list = code_variables.split(',')
    vocab_size = tokenizer.vocab_size

    doc_tokens = tokenizer.tokenize(doc)[:args.max_nl_len]

    if args.model_type == 'codebert' or args.model_type == 'unixcoder':
        var_tokens = ([tokenizer.cls_token] + doc_tokens + [tokenizer.sep_token] +
                      tokenizer.tokenize(code_mask_var))[:args.max_seq_len - 1] + [tokenizer.eos_token]
        str_tokens = ([tokenizer.cls_token] + doc_tokens + [tokenizer.sep_token] +
                      tokenizer.tokenize(code_mask_str))[:args.max_seq_len - 1] + [tokenizer.eos_token]
        token_type_ids = ([0] * (len(doc_tokens) + 2) + [1] * args.max_seq_len)[:args.max_seq_len]
    else:
        var_tokens = (doc_tokens + [tokenizer.sep_token] + tokenizer.tokenize(code_mask_var))[:args.max_seq_len]
        str_tokens = (doc_tokens + [tokenizer.sep_token] + tokenizer.tokenize(code_mask_str))[:args.max_seq_len]
        token_type_ids = ([0] * (len(doc_tokens) + 1) + [1] * args.max_seq_len)[:args.max_seq_len]
    var_ids = tokenizer.convert_tokens_to_ids(var_tokens)
    var_mask = [1] * (len(var_tokens))
    padding_length = args.max_seq_len - len(var_ids)
    var_ids += [tokenizer.pad_token_id] * padding_length
    var_mask += [0] * padding_length

    str_ids = tokenizer.convert_tokens_to_ids(str_tokens)
    str_mask = [1] * (len(str_tokens))
    padding_length = args.max_seq_len - len(str_ids)
    str_ids += [tokenizer.pad_token_id] * padding_length
    str_mask += [0] * padding_length

    input_ids = str_ids
    attention_mask = str_mask

    if args.model_type == 'codebert' or args.model_type == 'unixcoder':
        label_tokens = str_tokens[:]
    else:
        if len(str_tokens) == args.max_seq_len:
            label_tokens = str_tokens[: len(str_tokens) - 1] + [tokenizer.eos_token]
        else:
            label_tokens = str_tokens + [tokenizer.eos_token]
    label = tokenizer.convert_tokens_to_ids(label_tokens)
    padding_length = args.max_seq_len - len(label_tokens)
    label += [args.ignore_index] * padding_length

    if tokenizer.sep_token_id in input_ids:
        sep_index = input_ids.index(tokenizer.sep_token_id)
    else:
        sep_index = 0

    c, m, v = sep_index + 1, sep_index + 1, 0
    for ignore_str_index in range(sep_index + 1):
        label[ignore_str_index] = args.ignore_index

    err_flag = False
    local_var = var_list[v]

    if len(code_variables) > 0:
        while c < len(str_tokens):
            if input_ids[c] == tokenizer.pad_token_id:
                break
            # contents are the same, move to the next token for both code tokens
            if var_tokens[m] == str_tokens[c] or (special_split_symbol + var_tokens[m]) == str_tokens[c]:
                label[c] = args.ignore_index
                c += 1
                m += 1

            elif (special_split_symbol + var_tokens[m]).startswith(str_tokens[c]):
                var_tokens[m] = var_tokens[m][len(str_tokens[c].replace(special_split_symbol, '')):]
                label[c] = args.ignore_index
                c += 1

            elif var_tokens[m].startswith(str_tokens[c]):
                var_tokens[m] = var_tokens[m][len(str_tokens[c]):]
                label[c] = args.ignore_index
                c += 1

            elif str_tokens[c].startswith((special_split_symbol + var_tokens[m])):
                str_tokens[c] = str_tokens[c][len(special_split_symbol + var_tokens[m]):]
                m += 1

            elif str_tokens[c].startswith((var_tokens[m])):
                str_tokens[c] = str_tokens[c][len(var_tokens[m]):]
                m += 1

            # not equal means the token is a variable, need to be masked
            else:
                # masked code tokens move to the next token that appears in raw code tokens
                # these can not exist two continues variables, thus m += 1 means move to the next token and
                # skip the special masked token
                # remove blank space of local variables
                try:
                    local_var = local_var.replace(' ', '')
                    while local_var:
                        assert local_var.startswith(str_tokens[c]) or (special_split_symbol + local_var).startswith(
                            str_tokens[c]), "error"
                        if local_var.startswith(str_tokens[c]):
                            local_var = local_var[len(str_tokens[c]):]
                        else:
                            local_var = local_var[len(str_tokens[c].replace(special_split_symbol, '')):]

                        # add random mask
                        # 80% mask
                        # 10% keep the same token
                        # 10% replace with another token
                        # the setting is the same with BERT
                        rand = random.random()
                        if rand > 0.2:
                            attention_mask[c] = 0
                            input_ids[c] = tokenizer.mask_token_id
                        elif rand < 0.1:
                            random_id = random.randint(0, vocab_size - 1)
                            input_ids[c] = random_id
                        else:
                            input_ids[c] = input_ids[c]
                        c += 1
                        if c == len(str_tokens):  # code has been truncated
                            break
                except (AssertionError, IndexError):
                    err_flag = True
                    break

                v += 1
                if v == len(var_list):
                    while c < len(str_tokens):
                        label[c] = args.ignore_index
                        c += 1
                    break
                local_var = var_list[v]

            if m == len(var_tokens) or c == len(str_tokens):
                break
    else:
        # keep the same with bert
        for c in range(len(str_tokens)):
            rand = random.random()
            if rand < 0.15:
                rand /= 0.15

                if rand > 0.2:
                    attention_mask[c] = 0
                    input_ids[c] = tokenizer.mask_token_id
                elif rand < 0.1:
                    random_id = random.randint(0, vocab_size - 1)
                    input_ids[c] = random_id
                else:
                    input_ids[c] = input_ids[c]
            else:
                label[c] = args.ignore_index

    if not err_flag:
        assert len(token_type_ids) == len(input_ids) == len(attention_mask) == len(
            label) == args.max_seq_len, f"embedding error"
        source_ids = input_ids
        source_token_type_id = token_type_ids
        source_mask = attention_mask
        target_ids = label

        tgt_ids = []
        for i in label:
            if i >= 0:
                tgt_ids.append(i)

        input_token = tokenizer.convert_ids_to_tokens(source_ids, skip_special_tokens=False)
        target_token = tokenizer.convert_ids_to_tokens(tgt_ids)
        if len(target_token) == 0:
            return None, None, None, None
        return source_ids, source_token_type_id, source_mask, target_ids
    else:
        return None, None, None, None


def data_collator_fn(batch, tokenizer, args):
    docs, langs, codes_mask_str, codes_strings, \
    codes_mask_var, codes_mask_func_call, codes_mask_identifier, \
    codes_variables, codes_func_calls, codes_identifiers = map(list, zip(*batch))

    input_ids = []
    token_type_ids = []
    attention_masks = []
    labels = []

    err_num = 0
    for (doc, lang, code_mask_str, code_strings,
         code_mask_var, code_mask_func_call, code_mask_identifier,
         code_variables, code_func_calls, code_identifiers) in \
            zip(docs, langs, codes_mask_str, codes_strings,
                codes_mask_var, codes_mask_func_call, codes_mask_identifier,
                codes_variables, codes_func_calls, codes_identifiers):
        for token in tokenizer.all_special_tokens:
            doc = doc.replace(token, " ")
            code_mask_str = code_mask_str.replace(token, " ")
            code_mask_var = code_mask_var.replace(token, " ")
            code_mask_func_call = code_mask_func_call.replace(token, " ")
            code_mask_identifier = code_mask_identifier.replace(token, " ")

        if args.mask_type == 'const':
            input_id, token_type_id, attention_mask, label = \
                mask_constant(doc, lang, code_mask_str, code_strings, code_mask_identifier, code_identifiers, tokenizer, args)
        elif args.mask_type == 'str':
            input_id, token_type_id, attention_mask, label = \
                mask_string(doc, lang, code_mask_str, code_strings, tokenizer, args)
        elif args.mask_type == 'var':
            input_id, token_type_id, attention_mask, label = \
                mask_nonrandom(doc, lang, code_mask_str, code_strings, code_mask_var, code_variables, tokenizer, args)
        elif args.mask_type == 'func':
            input_id, token_type_id, attention_mask, label = \
                mask_nonrandom(doc, lang, code_mask_str, code_strings, code_mask_func_call, code_func_calls, tokenizer, args)
        elif args.mask_type == 'iden':
            input_id, token_type_id, attention_mask, label = \
                mask_nonrandom(doc, lang, code_mask_str, code_strings, code_mask_identifier, code_identifiers, tokenizer, args)
        else:
            assert args.mask_type == 'rand', f"mask type error"
            input_id, token_type_id, attention_mask, label = \
                mask_random(doc, lang, code_mask_str, code_strings, tokenizer, args)

        if input_id is not None:
            input_ids.append(input_id)
            token_type_ids.append(token_type_id)
            attention_masks.append(attention_mask)
            labels.append(label)
        else:
            err_num += 1

    while err_num > 0:
        rand_index = random.randint(0, len(input_ids) - 1)
        input_ids.append(input_ids[rand_index])
        attention_masks.append(attention_masks[rand_index])
        token_type_ids.append(token_type_ids[rand_index])
        labels.append(labels[rand_index])
        err_num -= 1

    input_ids = torch.LongTensor(input_ids)
    attention_masks = torch.LongTensor(attention_masks)
    token_type_ids = torch.LongTensor(token_type_ids)
    labels = torch.LongTensor(labels)

    # logging.info(input_ids.shape)
    # logging.info(attention_masks.shape)
    # logging.info(token_type_ids.shape)
    # logging.info(labels.shape)
    # logging.info(torch.Size([args.train_batch_size, args.max_seq_len]))

    assert input_ids.shape == attention_masks.shape == token_type_ids.shape == \
           labels.shape, f"batch embedding error"

    # == torch.Size([args.train_batch_size, args.max_seq_len]) for multi-gpu

    batch_dict = {
        'input_ids': input_ids,
        'attention_mask': attention_masks,

        'labels': labels,
    }

    return batch_dict
