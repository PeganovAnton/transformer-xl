import collections
import glob
import itertools
import json
import os
import random
from argparse import ArgumentParser
from typing import Tuple, Iterable

import tqdm

MIN_LENGTH = 5

PROJECT_SPLIT_SYMBOL = "\n龖龖龖\n"
FILE_SPLIT_SYMBOL = "\n!龖!\n"

TRAIN_FILE_SIZE = 100 * (1000 ** 2)  # Convert 100MB into bytes
VAL_FILE_SIZE = 1 * (1000 ** 2)
TEST_FILE_SIZE = 1 * (1000 ** 2)

SEED = None


def prepare_project(project: Iterable[Tuple[str, str]], cur_file: Tuple[str, str] = None) -> str:
    def filter_project(project: Iterable[Tuple[str, str]]) -> Iterable[Tuple[str, str]]:
        def filename_is_good(filename: str) -> bool:
            return True

        def content_is_good(content: str) -> bool:
            if len(content) < MIN_LENGTH:
                return False
            return True

        return (
            (filename, content)
            for filename, content in project
            if filename_is_good(filename) and content_is_good(content)
        )

    def preprocess_project(project: Iterable[Tuple[str, str]]) -> Iterable[Tuple[str, str]]:
        def preprocess_filename(filename: str) -> str:
            return filename

        def preprocess_content(content: str) -> str:
            return content

        return ((preprocess_filename(filename), preprocess_content(content)) for filename, content in project)

    def project_to_string(project: Iterable[Tuple[str, str]]) -> str:
        def merge_path_content(path: str, content: str) -> str:
            return f"{FILE_SPLIT_SYMBOL}<<!<<{path}>>!>>\n{content}"

        text = PROJECT_SPLIT_SYMBOL
        text += "".join(merge_path_content(path, content) for path, content in project)
        return text

    project = itertools.chain(project, (cur_file,) if cur_file else tuple())
    project = filter_project(project)
    project = preprocess_project(project)
    project = project_to_string(project)
    return project


def write_file(file_path: str, texts: Iterable[str], file_size: int = None) -> bool:
    if not file_size:
        file_size = float("+inf")
    file_object = open(file_path, "wt", encoding="utf-8")
    cur_file_size = 0

    for text in texts:
        cur_file_size += file_object.write(text)

        if cur_file_size >= file_size:
            file_object.close()
            return True
    file_object.close()
    return False


def get_projects_v3(raw_data_path: str, languages: Iterable[str] = ("Python",)) -> Iterable[Iterable[Tuple[str, str]]]:
    def read_file(path):
        with open(path, "rt", encoding="utf-8", errors="ignore") as f:
            return f.read()

    random.seed(SEED)

    files = []
    for language in languages:
        files += glob.iglob(os.path.join(raw_data_path, "v3", "languages", language, ".*", "**", "*.*"), recursive=True)
    assert files, f"There are no files in {raw_data_path} with languages {languages}"

    projects = collections.defaultdict(list)
    for file in files:
        if os.path.isfile(file):
            project_name = "/".join(file.split("/")[-3:-1])
            projects[project_name].append(file)

    projects = list(projects.items())
    random.shuffle(projects)
    print(f"Found {len(projects)} projects")

    total_bar = tqdm.tqdm()
    total_bar.set_description("Projects was processed")

    for project_name, files in projects:
        paths_dict_path = os.path.join(
            raw_data_path,
            "v3",
            "repositories",
            files[0].split("/")[5],
            files[0].split("/")[6],
            files[0].split("/")[7],
            "paths.json",
        )
        with open(paths_dict_path, "rt") as f:
            paths_dict = json.load(f)
        paths_and_contents = (
            (paths_dict[os.path.basename(file)], read_file(file))
            for file in files
            if os.path.basename(file) in paths_dict
        )
        total_bar.update(1)
        yield paths_and_contents


def main(raw_data_path: str, dataset_path: str) -> None:
    def get_train_file_paths(parent_path: str) -> Iterable[str]:
        return (os.path.join(parent_path, f"git_{i}.txt") for i in itertools.count(1))

    def get_valid_file_paths(parent_path: str) -> str:
        return os.path.join(parent_path, f"valid.txt")

    def get_test_file_paths(parent_path: str) -> str:
        return os.path.join(parent_path, f"test.txt")

    projects = get_projects_v3(raw_data_path)

    projects = map(prepare_project, projects)

    # Write to valid
    write_file(get_valid_file_paths(dataset_path), projects, VAL_FILE_SIZE)

    # Write to test
    write_file(get_test_file_paths(dataset_path), projects, TEST_FILE_SIZE)

    # Write to train
    train_paths = get_train_file_paths(dataset_path)

    while write_file(next(train_paths), projects, TRAIN_FILE_SIZE):
        pass


if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument("--raw_data_path", type=str, required=True, help="path to the dataset (v3) folder")
    args.add_argument("--dataset_path", type=str, required=True, help="where to store train valid test files")
    args.add_argument("--min_len", type=int, default=5, help="minimum file content length")
    args.add_argument("--seed", type=int, default=None, help="random seed for example shuffle")
    args = args.parse_args()

    MIN_LENGTH = args.min_len
    SEED = args.seed

    main(args.raw_data_path, args.dataset_path)
