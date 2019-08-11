import glob
import itertools
import os
import random
from argparse import ArgumentParser
from collections import defaultdict
from typing import Tuple, Generator, Iterator, List, TextIO

MIN_LENGTH = 5
PROJECT_SPLIT_SYMBOL = "\n龖龖龖\n"
EXAMPLE_SPLIT_SYMBOL = "\n!龖!\n"
TRAIN_FILE_SIZE = 100 * (1000 ** 2)  # Convert 100MB into bytes
VAL_FILE_SIZE = 1 * (1000 ** 2)
TEST_FILE_SIZE = 1 * (1000 ** 2)
SEED = None


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


def get_path_and_content(path: str) -> Tuple[str, str]:
    filename = os.path.basename(path).split("_")
    with open(path, "rt", encoding="utf-8", errors="ignore") as f:
        content = f.read()
    # project_name = filename[1]
    example_path = "/".join(filename[2:])
    return example_path, content


def merge_path_content(path: str, content: str) -> str:
    return f"{EXAMPLE_SPLIT_SYMBOL}<<!<<{path}>>!>>\n{content}"


def get_train_file_paths(parent_path: str) -> Generator[str, None, None]:
    return (os.path.join(parent_path, f"git_{i}.txt") for i in itertools.count(1))


def get_valid_file_paths(parent_path: str) -> str:
    return os.path.join(parent_path, f"valid.txt")


def get_test_file_paths(parent_path: str) -> str:
    return os.path.join(parent_path, f"test.txt")


def close_file(file_object: TextIO) -> None:
    # file_object.write(PROJECT_SPLIT_SYMBOL)
    file_object.close()


def write_examples(file_path: str, projects: Iterator[Tuple[str, Iterator[str]]], min_file_size: int) -> bool:
    file_object = open(file_path, "wt", encoding="utf-8")
    cur_file_size = 0
    for project in projects:
        cur_file_size += file_object.write(PROJECT_SPLIT_SYMBOL)
        project_name, example_paths = project
        for example_path in example_paths:
            filename, content = get_path_and_content(example_path)
            if filename_is_good(filename) and content_is_good(content):
                filename, content = filter_filename(filename), filter_content(content)
                to_write = merge_path_content(filename, content)
                cur_file_size += file_object.write(to_write)

        if cur_file_size >= min_file_size:
            close_file(file_object)
            return True
    close_file(file_object)
    return False


def get_examples_iterator(file_paths: List[str]) -> Iterator[Tuple[str, Iterator[str]]]:
    # Group by project
    projects = defaultdict(list)
    for file_path in file_paths:
        project_name = os.path.basename(file_path).split("_")[1]
        projects[project_name].append(file_path)
    projects = list(projects.items())
    # Shuffle
    random.seed(SEED)
    random.shuffle(projects)
    for project in projects:
        random.shuffle(project[1])

    return iter(map(lambda pair: (pair[0], iter(pair[1])), projects))


def main(file_path_pattern: str, dataset_path: str) -> None:
    source_file_paths = glob.glob(file_path_pattern)
    assert source_file_paths, f"Nothing found at {file_path_pattern}"
    examples = get_examples_iterator(source_file_paths)

    # Write to valid
    write_examples(get_valid_file_paths(dataset_path), examples, VAL_FILE_SIZE)

    # Write to test
    write_examples(get_test_file_paths(dataset_path), examples, TEST_FILE_SIZE)

    # Write to train
    train_paths = get_train_file_paths(dataset_path)
    while write_examples(next(train_paths), examples, TRAIN_FILE_SIZE):
        pass


if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument("--pattern", type=str, required=True, help="git file paths pattern")
    args.add_argument("--dataset_path", type=str, required=True, help="where to store train valid test files")
    args.add_argument("--min_len", type=int, default=5, help="minimum file content length")
    args.add_argument("--seed", type=int, default=None, help="random seed for example shuffle")
    args = args.parse_args()

    MIN_LENGTH = args.min_len
    SEED = args.seed

    main(args.pattern, args.dataset_path)
