"""Filter out short files and delete non utf-8 bytes"""
import glob
import itertools
import os
from argparse import ArgumentParser
from random import shuffle
from typing import Tuple, Generator

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


def get_filename_content(path: str) -> Tuple[str, str]:
    filename = os.path.basename(path)
    with open(path, "rt", encoding="utf-8", errors="ignore") as f:
        content = f.read()
    return filename, content


def merge_filename_content(filename: str, content: str) -> str:
    return f"<<!<<{filename}>>!>>\n{content}{SPLIT_SYMBOL}"


def get_train_file_paths(parent_path: str) -> Generator[str, None, None]:
    return (os.path.join(parent_path, f"git_{i}.txt") for i in itertools.count(1))


def get_valid_file_paths(parent_path: str) -> str:
    return os.path.join(parent_path, f"valid.txt")


def get_test_file_paths(parent_path: str) -> str:
    return os.path.join(parent_path, f"test.txt")


def main(file_path_pattern: str, dataset_path: str) -> None:
    source_file_paths = glob.glob(file_path_pattern)
    assert source_file_paths, f"Nothing found at {file_path_pattern}"
    shuffle(source_file_paths)
    source_file_paths, files_total_len = iter(source_file_paths), len(source_file_paths)

    train_paths = get_train_file_paths(dataset_path)
    valid_file = open(get_valid_file_paths(dataset_path), "wt", encoding="utf-8")
    test_file = open(get_test_file_paths(dataset_path), "wt", encoding="utf-8")
    train_file = open(next(train_paths), "wt", encoding="utf-8")

    tr_count, all_count = 0, 0
    train_files_count = 1
    cur_train_size, cur_test_size, cur_valid_size, all_size = 0, 0, 0, 0
    min_size, max_size = float("+inf"), float("-inf")

    # Write to valid
    for file_path in source_file_paths:
        all_count += 1
        filename, content = get_filename_content(file_path)
        if filename_is_good(filename) and content_is_good(content):
            filename, content = filter_filename(filename), filter_content(content)
            filename_content = merge_filename_content(filename, content)
            len_content = len(content)
            all_size += len_content

            valid_file.write(filename_content)
            cur_valid_size += len_content

            if cur_valid_size >= VAL_FILE_SIZE:
                valid_file.close()
                break

    # Write to test
    for file_path in source_file_paths:
        all_count += 1
        filename, content = get_filename_content(file_path)
        if filename_is_good(filename) and content_is_good(content):
            filename, content = filter_filename(filename), filter_content(content)
            filename_content = merge_filename_content(filename, content)
            len_content = len(content)
            all_size += len_content

            test_file.write(filename_content)
            cur_test_size += len_content

            if cur_test_size >= TEST_FILE_SIZE:
                test_file.close()
                break

    # Write to train
    for file_path in tqdm(source_file_paths, total=files_total_len - all_count):
        all_count += 1
        filename, content = get_filename_content(file_path)
        if filename_is_good(filename) and content_is_good(content):
            tr_count += 1
            filename = filter_filename(filename)
            content = filter_content(content)
            filename_content = merge_filename_content(filename, content)
            len_content = len(content)
            all_size += len_content
            min_size, max_size = min(min_size, len_content), max(max_size, len_content)

            train_file.write(filename_content)
            cur_train_size += len_content

            if cur_train_size >= TRAIN_FILE_SIZE:  # assume that each symbol takes 1 byte
                cur_train_size = 0
                train_file.close()
                train_file = open(next(train_paths), "wt", encoding="utf-8")
                train_files_count += 1
    train_file.close()

    print(f"{tr_count} of {all_count} examples have been filtered and wrote to {train_files_count} files ~100MB each\n"
          f"min/average/max lengths of filtered examples are {min_size}/{all_size / tr_count}/{max_size} symbols")


if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument("--pattern", type=str, required=True, help="git file paths pattern")
    args.add_argument("--dataset_path", type=str, required=True, help="where to store train valid test files")
    args.add_argument("--min_len", type=int, default=5, help="minimum file content length")
    args = args.parse_args()

    MIN_LENGTH = args.min_len

    main(args.pattern, args.dataset_path)
