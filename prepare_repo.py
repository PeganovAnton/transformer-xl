import glob
import os
import random
from argparse import ArgumentParser
from typing import Tuple, Generator

import tqdm

from prepare_git_data import write_examples

MIN_LENGTH = 50

PROJECT_SPLIT_SYMBOL = "\n龖龖龖\n"
EXAMPLE_SPLIT_SYMBOL = "\n!龖!\n"

TRAIN_FILE_SIZE = 100 * (1000 ** 2)  # Convert 100MB into bytes
VAL_FILE_SIZE = 1 * (1000 ** 2)
TEST_FILE_SIZE = 1 * (1000 ** 2)

SHUFFLE = False
SEED = None


def get_examples_generator(repo_path: str) -> Generator[Tuple[str, str], None, None]:
    source_file_paths = glob.glob(os.path.join(repo_path, "**/*.py"), recursive=True)

    if SHUFFLE:
        random.seed(SEED)
        random.shuffle(source_file_paths)

    progress_bar = tqdm.tqdm()
    progress_bar.set_description("Files was processed")

    for file_path in source_file_paths:
        relative_path = os.path.relpath(file_path, start=repo_path)
        with open(file_path, "rt", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        yield relative_path, content
        progress_bar.update(1)


def main(repo_path: str, result_file_path: str) -> None:
    examples = get_examples_generator(repo_path)
    write_examples(result_file_path, examples)


if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument(
        "--repo_path", type=str, required=True, help="path to repository, from which data will be generated"
    )
    args.add_argument(
        "--result_file_path", type=str, required=True, help="path to dataset file, where result will be stored"
    )
    args.add_argument("--min_len", type=int, default=50, help="minimum file content length")
    args.add_argument("--shuffle", action="store_true", help="shuffle files")
    args.add_argument("--seed", type=int, default=None, help="random seed for example shuffle")
    args = args.parse_args()

    MIN_LENGTH = args.min_len
    SHUFFLE = args.shuffle
    SEED = args.seed

    main(args.repo_path, args.result_file_path)
