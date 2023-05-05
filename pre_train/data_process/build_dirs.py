import os
from pre_train.utils import args


def make_need_dir(all_dir):
    for need_dir in all_dir:
        if not os.path.exists(need_dir):
            os.makedirs(need_dir)
            print(f'build data dirs: {need_dir}')


def clean_dir(dir_name, keep_file_type=None, keep_file_name=None):
    if os.path.isdir(dir_name):
        for file_name in os.listdir(dir_name):
            file = os.path.join(dir_name, file_name)
            if keep_file_type is not None:
                if not file.endswith(keep_file_type):
                    os.remove(file)
            else:
                os.remove(file)


def clean_old_data(lang_dirs):
    make_need_dir(lang_dirs)
    print(args.console_sep_string)
    print('start to clean old data')
    # clear dir
    for lang_dir in lang_dirs:
        for is_train in os.listdir(lang_dir):
            data_split_dir = os.path.join(lang_dir, is_train)
            if os.path.isdir(data_split_dir):
                clean_dir(data_split_dir, 'jsonl')
    print('clean data finished')
    print(args.console_sep_string)
