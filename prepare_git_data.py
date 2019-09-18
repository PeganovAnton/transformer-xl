import itertools
import os
import random
import tarfile
from argparse import ArgumentParser
from typing import Tuple, Generator

import tqdm

MIN_LENGTH = 5

PROJECT_SPLIT_SYMBOL = "\n龖龖龖\n"
EXAMPLE_SPLIT_SYMBOL = "\n!龖!\n"

TRAIN_FILE_SIZE = 100 * (1000 ** 2)  # Convert 100MB into bytes
VAL_FILE_SIZE = 1 * (1000 ** 2)
TEST_FILE_SIZE = 1 * (1000 ** 2)

SEED = None
FILES_TO_LOAD_AT_ONCE = 1000000


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


def get_project_relative_path(path: str) -> str:
    """Returns project relative path"""
    path = os.path.basename(path)
    if path.startswith("siva_"):
        path = path[5:]
    basename, ext = os.path.splitext(path)
    names = basename.split("_")
    # Drop time
    proj_rel_path = "/".join(names[:-1]) + ext

    return proj_rel_path


def merge_path_content(path: str, content: str) -> str:
    return f"{EXAMPLE_SPLIT_SYMBOL}<<!<<{path}>>!>>\n{content}"


def get_train_file_paths(parent_path: str) -> Generator[str, None, None]:
    return (os.path.join(parent_path, f"git_{i}.txt") for i in itertools.count(1))


def get_valid_file_paths(parent_path: str) -> str:
    return os.path.join(parent_path, f"valid.txt")


def get_test_file_paths(parent_path: str) -> str:
    return os.path.join(parent_path, f"test.txt")


def write_examples(file_path: str, examples: Generator[Tuple[str, str], None, None], file_size: int = None) -> bool:
    if not file_size:
        file_size = float("+inf")
    file_object = open(file_path, "wt", encoding="utf-8")
    cur_file_size = 0

    for filename, content in examples:
        if filename_is_good(filename) and content_is_good(content):
            filename, content = filter_filename(filename), filter_content(content)
            to_write = merge_path_content(filename, content)
            cur_file_size += file_object.write(to_write)

        if cur_file_size >= file_size:
            file_object.close()
            return True
    file_object.close()
    return False


def get_examples_generator(tar_path: str) -> Generator[Tuple[str, str], None, None]:
    random.seed(SEED)
    tar = tarfile.open(tar_path, mode="r:gz")

    progress_bar = tqdm.tqdm()
    progress_bar.set_description("Files was processed")

    time_to_stop = False
    while True:
        print(f"\nLoading next {FILES_TO_LOAD_AT_ONCE} files...")
        files_batch = []
        for _ in tqdm.trange(FILES_TO_LOAD_AT_ONCE):
            file = tar.next()
            if not file:
                time_to_stop = True
                break
            if not file.isfile():
                continue

            file_object = tar.extractfile(file)
            files_batch.append(
                (get_project_relative_path(file.name), file_object.read().decode(encoding="utf-8", errors="ignore"))
            )
            file_object.close()

        random.shuffle(files_batch)

        for file_path, content in files_batch:
            yield file_path, content
            progress_bar.update(1)

        if time_to_stop:
            break

    tar.close()


def main(tar_path: str, dataset_path: str) -> None:
    examples = get_examples_generator(tar_path)

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
    args.add_argument("--tar_path", type=str, required=True, help="path to tar with dataset")
    args.add_argument("--dataset_path", type=str, required=True, help="where to store train valid test files")
    args.add_argument("--min_len", type=int, default=5, help="minimum file content length")
    args.add_argument("--seed", type=int, default=None, help="random seed for example shuffle")
    args.add_argument("--files_in_memory", type=int, default=1000000, help="how many files store in memory to shuffle")
    args = args.parse_args()

    MIN_LENGTH = args.min_len
    SEED = args.seed
    FILES_TO_LOAD_AT_ONCE = args.files_in_memory

    main(args.tar_path, args.dataset_path)
