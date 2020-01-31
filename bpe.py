import datetime
import os
import pickle
import sys
import time
from glob import glob
from multiprocessing.pool import Pool
from typing import List, Iterable, Union, Optional, Collection

import numpy as np
import pandas as pd
import youtokentome as yttm
from pygments.lexers.python import Python3Lexer
from pygments.token import is_token_subtype, Comment, String, Text
from tqdm import tqdm
from transformers import GPT2Tokenizer
from youtokentome import OutputType


class GitBPE(object):
    _available_methods = {"English", "PyToken", "Line"}

    _spaces = [" ", "\n", "\t", "\r"]
    _space_to_special_splits = {" ": " ·", "\n": " ⍽ ", "\r": "", "\t": " ␣ "}  # Skip rare \f and \v symbols
    _space_to_special_nosplits = {" ": "·", "\n": "⍽", "\r": "", "\t": "␣"}  # Skip rare \f and \v symbols
    _special_to_space = {"·": " ", "⍽": "\n", "␣": "\t"}  # Skip rare \f and \v symbols

    def __init__(self, model_path: str, method: str):
        assert (
            method in GitBPE._available_methods
        ), f"Invalid _method name: {method}. You can use any of these: {GitBPE._available_methods}"

        self._method = method

        self._trained_model: yttm.BPE = None
        self._trained_model_path = model_path
        self._py_lexer = Python3Lexer()

    @staticmethod
    def train(
        *,
        method: str,
        data_paths: Iterable[str],
        model_path: str,
        vocab_size: int,
        additional_symbols: Iterable[str] = tuple(),
        coverage: float = 1.0,
        n_threads: int = -1,
        pad_id: int = 0,
        unk_id: int = 1,
        bos_id: int = 2,
        eos_id: int = 3,
    ) -> "GitBPE":
        git_bpe = GitBPE(model_path, method)

        assert data_paths, f"You must provide at least one data path. data_paths: {data_paths}"
        data_path = git_bpe._preprocess_files(data_paths)
        if additional_symbols:
            with open(data_path, "a", encoding="utf-8") as f:
                f.write("\n" + "\n".join(additional_symbols))

        git_bpe._trained_model = yttm.BPE.train(
            data_path, model_path, vocab_size, coverage, n_threads, pad_id, unk_id, bos_id, eos_id
        )

        git_bpe._save()

        return git_bpe

    def encode(
        self,
        texts: List[str],
        output_type: OutputType = OutputType.ID,
        bos: bool = False,
        eos: bool = False,
        reverse: bool = False,
        dropout_prob: float = 0,
        preprocessed_texts: List[str] = None,
    ) -> Union[List[List[int]], List[List[str]]]:
        if preprocessed_texts:
            return self._trained_model.encode(preprocessed_texts, output_type, bos, eos, reverse, dropout_prob)
        preprocessed_texts = list(map(self._preprocess_text, texts))
        return self._trained_model.encode(preprocessed_texts, output_type, bos, eos, reverse, dropout_prob)

    def vocab_size(self) -> int:
        return self._trained_model.bpe_cython.vocab_size()

    def vocab(self) -> List[str]:
        return self._trained_model.bpe_cython.vocab()

    def subword_to_id(self, subword: str) -> int:
        return self._trained_model.bpe_cython.subword_to_id(subword)

    def id_to_subword(self, id: int) -> str:
        return self._trained_model.bpe_cython.id_to_subword(id)

    def decode(self, ids: List[List[int]], ignore_ids: Optional[Collection[int]] = None) -> List[str]:
        texts = self._trained_model.bpe_cython.decode(ids, ignore_ids)
        return [self._postprocess_text(text) for text in texts]

    def represent(self, text: str, sep_symbol: str = "|") -> str:
        subwords = self.encode([text], output_type=OutputType.SUBWORD)[0]
        subwords[0] = subwords[0][1:]
        text = sep_symbol.join(subwords)
        text = text.replace("▁", " ")
        text = self._postprocess_text(text)
        return text

    def _save(self) -> None:
        path = self._trained_model_path + "-gitbpe.pkl"
        trained_model = self._trained_model
        self._trained_model = None
        self._py_lexer = None
        with open(path, "wb") as f:
            pickle.dump(self, f)
        self._trained_model = trained_model

    @staticmethod
    def load(path: str) -> "GitBPE":
        gitbpe_model_path = path + "-gitbpe.pkl"
        with open(gitbpe_model_path, "rb") as f:
            git_bpe = pickle.load(f)

        yttm_model_path = path
        git_bpe._trained_model = yttm.BPE(yttm_model_path)
        git_bpe._py_lexer = Python3Lexer()

        return git_bpe

    def _preprocess_files(self, filepaths: Iterable[str]) -> str:
        # Path to tmp file
        timestamp = datetime.datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d_%H:%M:%S")
        self._train_file = f"/tmp/git-yttm-{self._method}-{timestamp}"

        with open(self._train_file, "wt", encoding="utf-8") as output_file:
            for filepath in tqdm(filepaths, desc="Preparing train files..."):
                with open(filepath, "rt", encoding="utf-8", errors="ignore") as input_file:
                    lines = input_file.readlines()
                output_file.writelines(self._preprocess_text("".join(lines)))
        return self._train_file

    def _preprocess_text(self, text: str) -> str:
        if self._method == "English":
            text = self._replace_spaces(text)
            return text

        elif self._method == "PyToken":
            tokens = self._tokenize_code(text, language="Python3")

            preprocessed_tokens = []
            for _, token_type, token in tokens:
                if token in GitBPE._spaces:
                    preprocessed_token = GitBPE._space_to_special_splits[token]
                elif is_token_subtype(token_type, Text):  # Multiple tabs case and other strange cases
                    preprocessed_token = self._replace_spaces(token)
                elif is_token_subtype(token_type, Comment) or is_token_subtype(token_type, String):
                    preprocessed_token = self._replace_spaces(token)
                else:
                    preprocessed_token = token + " "
                preprocessed_tokens.append(preprocessed_token)

            return "".join(preprocessed_tokens)

        else:
            assert self._method == "Line"
            text = GitBPE._replace_spaces(text, spaces=" \t\r", with_splits=False)
            text = GitBPE._replace_spaces(text, spaces="\n")
            return text

    def _postprocess_text(self, text: str) -> str:
        if self._method == "English":
            text = text.replace(" ", "")
            return self._replace_specials(text)
        elif self._method == "PyToken":
            text = text.replace(" ", "")
            return GitBPE._replace_specials(text)
        elif self._method == "Line":
            text = text.replace(" ", "")
            return GitBPE._replace_specials(text)

    @staticmethod
    def _replace_spaces(text: str, spaces: str = " \n\t\r", with_splits: bool = True) -> str:
        space_to_spec_dict = GitBPE._space_to_special_splits if with_splits else GitBPE._space_to_special_nosplits
        for space_symbol, special_symbol in space_to_spec_dict.items():
            if space_symbol in spaces:
                text = text.replace(space_symbol, special_symbol)
        return text

    @staticmethod
    def _replace_specials(text: str, spaces: str = " \n\t\r") -> str:
        for special_symbol, space_symbol in GitBPE._special_to_space.items():
            if space_symbol in spaces:
                text = text.replace(special_symbol, space_symbol)
        return text

    def _tokenize_code(self, code: str, language: str = "Python3") -> Iterable[str]:
        if language == "Python3":
            tokens = self._py_lexer.get_tokens_unprocessed(code)
            return tokens
        else:
            raise NotImplementedError
