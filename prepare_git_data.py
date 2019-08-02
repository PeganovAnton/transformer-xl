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
    return f"<<!<<{filename}>>!>>\n{content}"


def get_file_paths(parent_path: str) -> Generator[str, None, None]:
    return (os.path.join(parent_path, f"git_{i}.txt") for i in itertools.count(1))


def main(file_path_pattern: str, dataset_path: str) -> None:
    source_file_paths = glob.glob(file_path_pattern)
    shuffle(source_file_paths)
    assert source_file_paths, f"Nothing found at {file_path_pattern}"

    train_paths = get_file_paths(dataset_path)
    train_file = open(next(train_paths), "wt", encoding="utf-8")

    count = 0
    all_count = 0
    train_files_count = 1
    cur_size = 0
    all_size = 0
    max_size = float("-inf")
    min_size = float("+inf")
    for file_path in tqdm(source_file_paths):
        all_count += 1
        filename, content = get_filename_content(file_path)
        if filename_is_good(filename) and content_is_good(content):
            count += 1
            filename = filter_filename(filename)
            content = filter_content(content)
            filename_content = merge_filename_content(filename, content)

            train_file.write(filename_content)
            train_file.write(SPLIT_SYMBOL)

            len_content = len(content)
            cur_size += len_content
            all_size += len_content
            max_size = max(max_size, len_content)
            min_size = min(min_size, len_content)
            if cur_size >= TRAIN_FILE_SIZE:  # assume that each symbol takes 1 byte
                cur_size = 0
                train_file.close()
                train_file = open(next(train_paths), "wt", encoding="utf-8")
                train_files_count += 1
    train_file.close()

    print(f"{count} of {all_count} examples have been filtered and wrote to {train_files_count} files ~100MB each\n"
          f"min/average/max lengths of filtered examples are {min_size}/{all_size / count}/{max_size} symbols")


if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument("--pattern", type=str, required=True, help="git file paths pattern")
    args.add_argument("--dataset_path", type=str, required=True, help="where to store train valid test files")
    args.add_argument("--min_len", type=int, default=5, help="minimum file content length")
    args = args.parse_args()

    MIN_LENGTH = args.min_len

    main(args.pattern, args.dataset_path)
