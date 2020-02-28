import os
import pickle
from typing import List

import torch
from torch.utils.data import Dataset

from data_preprocessing.bpe import GitBPE
from hf_training.log import logger


class TextDataset(Dataset):
    def __init__(
        self,
        tokenizer: GitBPE,
        args,
        file_path: str,
        block_size=512,
        start_tokens: List[int] = None,
        dropout: float = 0,
    ):
        assert os.path.isfile(file_path)
        self._tokenizer = tokenizer

        block_size = block_size

        directory, filename = os.path.split(file_path)
        cached_features_file = os.path.join(directory, "_cached_lm_" + str(block_size) + "_" + filename)

        if os.path.exists(cached_features_file) and not args.overwrite_cache:
            logger.info("Loading features from cached file %s", cached_features_file)
            with open(cached_features_file, "rb") as handle:
                self.examples = pickle.load(handle)
        else:
            logger.info("Creating features from dataset file at %s", directory)

            self.examples = []
            with open(file_path, encoding="utf-8") as f:
                text = f.read()

            tokenized_text = tokenizer.encode([text], dropout_prob=dropout)[0]
            del text

            if start_tokens:
                logger.info(f"Calculating examples which start only with tokens: {start_tokens}")
                start_ids_inds = [token in start_tokens for i, token in enumerate(tokenized_text)]

            i = 0
            skipped_tokens = 0
            while i <= len(tokenized_text) - block_size:
                if start_tokens:
                    try:
                        offset = start_ids_inds.index(True, i) - i
                    except ValueError:
                        break
                    skipped_tokens += offset
                    i += offset

                example = tokenized_text[i : i + block_size]
                self.examples.append(example)
                i += block_size
            skipped_tokens += len(tokenized_text) - i - 1

            logger.info(f"Skipped {skipped_tokens} tokens")
            # Note that we are loosing the last truncated example here for the sake of simplicity (no padding)
            # If your dataset is small, first you should loook for a bigger one :-) and second you
            # can change this behavior by adding (model specific) padding.

            logger.info("Saving features into cached file %s", cached_features_file)
            with open(cached_features_file, "wb") as handle:
                pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        # example = self.examples[item]
        # print(self._tokenizer.decode([example])[0])
        return torch.tensor(self.examples[item], dtype=torch.long)


def load_and_cache_examples(args, tokenizer, evaluate=False):
    file_path = args.eval_data_file if evaluate else args.train_data_file
    return TextDataset(
        tokenizer,
        args,
        file_path=file_path,
        block_size=args.block_size,
        start_tokens=tokenizer.ids_with_sep,
        dropout=0 if evaluate else args.bpe_dropout,
    )
