"""Filter out short files and delete non utf-8 bytes"""
import glob
import itertools
import os
from argparse import ArgumentParser
from random import shuffle
from typing import Tuple, Generator

import typing
from tqdm import tqdm

MIN_LENGTH = 5
SPLIT_SYMBOL = "\n龖龖龖\n"
TRAIN_FILE_SIZE = 100 * (1000 ** 2)  # Convert 100MB into bytes
VAL_FILE_SIZE = 1 * (1000 ** 2)
TEST_FILE_SIZE = 1 * (1000 ** 2)


def filename_is_good(filename: str) -> bool:
    return True


def content_is_good(content: str) -> bool:
    if len(content) < MIN_LENGTH:
        return False
    return True


def filter_filename(filename: str) -> str:
    return filename


def filter_content(content: str) -> str:
    return content


def get_filename_and_content(path: str) -> Tuple[str, str]:
    filename = os.path.basename(path).split('_')
    with open(path, "rt", encoding="utf-8", errors="ignore") as f:
        content = f.read()
    repo_name = filename[1]
    example_path = '/'.join(filename[2:])
    return ' '.join((repo_name, example_path)), content


def merge_filename_content(filename: str, content: str) -> str:
    return f"{SPLIT_SYMBOL}<<!<<{filename}>>!>>\n{content}"


def get_train_file_paths(parent_path: str) -> Generator[str, None, None]:
    return (os.path.join(parent_path, f"git_{i}.txt") for i in itertools.count(1))


def get_valid_file_paths(parent_path: str) -> str:
    return os.path.join(parent_path, f"valid.txt")


def get_test_file_paths(parent_path: str) -> str:
    return os.path.join(parent_path, f"test.txt")


def close_file(file_object: typing.TextIO) -> None:
    file_object.write(SPLIT_SYMBOL)
    file_object.close()


def write_examples(file_path: str, examples_iterator: iter, file_size: int) -> bool:
    file_object = open(file_path, "wt", encoding="utf-8")
    cur_file_size = 0
    for example_path in examples_iterator:
        filename, content = get_filename_and_content(example_path)
        if filename_is_good(filename) and content_is_good(content):
            filename, content = filter_filename(filename), filter_content(content)
            to_write = merge_filename_content(filename, content)
            len_content = len(content)
            cur_file_size += len_content

            file_object.write(to_write)
            cur_file_size += len_content

            if cur_file_size >= file_size:
                close_file(file_object)
                return True
    close_file(file_object)
    return False


def main(file_path_pattern: str, dataset_path: str) -> None:
    source_file_paths = glob.glob(file_path_pattern)
    assert source_file_paths, f"Nothing found at {file_path_pattern}"
    shuffle(source_file_paths)
    source_file_paths, files_total_len = iter(source_file_paths), len(source_file_paths)

    train_paths = get_train_file_paths(dataset_path)

    # Write to valid
    write_examples(get_valid_file_paths(dataset_path), source_file_paths, VAL_FILE_SIZE)

    # Write to test
    write_examples(get_test_file_paths(dataset_path), source_file_paths, TEST_FILE_SIZE)

    # Write to train
    while write_examples(next(train_paths), source_file_paths, TRAIN_FILE_SIZE):
        pass


if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument("--pattern", type=str, required=True, help="git file paths pattern")
    args.add_argument("--dataset_path", type=str, required=True, help="where to store train valid test files")
    args.add_argument("--min_len", type=int, default=5, help="minimum file content length")
    args = args.parse_args()

    MIN_LENGTH = args.min_len

    main(args.pattern, args.dataset_path)
