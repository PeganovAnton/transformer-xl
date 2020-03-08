import itertools
import os
import pickle
import random
from typing import List, Tuple, Any

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, RandomSampler, DistributedSampler, SequentialSampler, DataLoader
from tqdm import tqdm

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
        tensor = torch.tensor(self.examples[item], dtype=torch.long)
        return tensor


class LMIterator:
    def __init__(
        self,
        file_path: str,
        tokenizer: GitBPE,
        batch_size: int,
        batch_len: int,
        start_tokens: Tuple[int] = (),
        dropout_prob: float = 0,
        overwrite_cache: bool = False,
        shuffle: bool = True,
        batch_first: bool = False,
    ):
        self._batch_size = batch_size
        self._batch_len = batch_len
        self._shuffle = shuffle
        self._batch_sequences = None
        self._need_transpose = not batch_first

        self._need_to_reshuffle = False

        directory, filename = os.path.split(file_path)
        cached_features_file = os.path.join(
            directory, "_cached_lm_mem_" + str(batch_size) + "_" + str(batch_len) + "_" + filename
        )

        if os.path.exists(cached_features_file) and not overwrite_cache:
            logger.info("Loading features from cached file %s", cached_features_file)
            with open(cached_features_file, "rb") as handle:
                self._examples = pickle.load(handle)
        else:
            logger.info("Creating features from dataset file at %s", directory)

            self.batches = []
            with open(file_path, encoding="utf-8") as f:
                text = f.read()

            logger.info(f"Tokenizing text...")
            tokenized_text = tokenizer.encode([text], dropout_prob=dropout_prob)[0]
            del text

            logger.info(f"Calculating examples which start only with tokens: {start_tokens}")
            self._examples = LMIterator.split_list(tokenized_text, start_tokens)
            self._examples = sorted(self._examples, key=lambda example: len(example), reverse=True)

            logger.info("Saving features into cached file %s", cached_features_file)
            with open(cached_features_file, "wb") as handle:
                pickle.dump(self._examples, handle, protocol=pickle.HIGHEST_PROTOCOL)

        self._form_batches()

    @staticmethod
    def split_list(a: list, split_values: Tuple[Any]) -> List[list]:
        start_ids = [i for i, val in tqdm(enumerate(a), "Splitting data to examples...") if val in split_values]
        assert start_ids[0] == 0 or start_ids[0] == 1
        result = [a[i:j] for i, j in zip(start_ids, start_ids[1:] + [len(a)])]
        assert sum(len(res) for res in result) <= len(a)
        return result

    def _form_batches(self):
        total_len = sum(len(example) for example in self._examples)
        batch_sequence_len = total_len // self._batch_size

        need_to_shuffle = self._shuffle and self._need_to_reshuffle
        if self._batch_sequences is None or need_to_shuffle:
            batch_sequences = [[] for _ in range(self._batch_size)]
            batch_sequence_lengths = [0 for _ in range(self._batch_size)]

            # Randomly distribute examples into batch sequences until they full
            # In init, we sorted examples such that long goes first
            for example in tqdm(self._examples, desc="Shuffling examples..."):
                cur_batch_sequence = random.randint(0, self._batch_size - 1)
                # If current is full, goes to the previous one
                while batch_sequence_lengths[cur_batch_sequence] >= batch_sequence_len:
                    cur_batch_sequence = (cur_batch_sequence - 1) % self._batch_size

                batch_sequences[cur_batch_sequence].append(example)
                batch_sequence_lengths[cur_batch_sequence] += len(example)

            # Shuffle each batch position to avoid example length bias
            for batch_sequence in tqdm(batch_sequences, desc="Shuffling batch sequences..."):
                random.shuffle(batch_sequence)

            # Concatenate examples
            batch_sequences = [
                list(itertools.chain(*batch_sequence))  # Fast analog to sum(batch_sequence, [])
                for batch_sequence in tqdm(batch_sequences, desc="Concatenating batch sequences...")
            ]

            min_batch_sequence_len = min(len(batch_sequence) for batch_sequence in batch_sequences)
            batch_sequences = [batch_sequence[:min_batch_sequence_len] for batch_sequence in batch_sequences]

            new_total_len = sum(len(batch_sequence) for batch_sequence in batch_sequences)
            logger.info(f"During the batch forming {new_total_len} of {total_len} tokens are left")

            self._batch_sequences = batch_sequences

    def __iter__(self):
        self._form_batches()

        for i in range(len(self)):
            data = [
                batch_sequence[i * self._batch_len : (i + 1) * self._batch_len]
                for batch_sequence in self._batch_sequences
            ]
            target = [
                batch_sequence[i * self._batch_len + 1 : (i + 1) * self._batch_len + 1]
                for batch_sequence in self._batch_sequences
            ]

            data = torch.tensor(data, dtype=torch.long)
            target = torch.tensor(target, dtype=torch.long)

            yield (data, target) if not self._need_transpose else (data.t(), target.t())

        self._need_to_reshuffle = True

    def __len__(self):
        num_batches, remain_tokens = divmod(len(self._batch_sequences[0]), self._batch_len)

        # Consider target shift
        if remain_tokens == 0:
            num_batches -= 1

        assert num_batches > 0

        return num_batches


def get_data_iterator(args, tokenizer: GitBPE, evaluate: bool = False):
    file_path = args.eval_data_file if evaluate else args.train_data_file

    if args.model_type == "gpt-2":

        def collate(examples: List[torch.Tensor]):
            return pad_sequence(examples, batch_first=True)

        dataset = TextDataset(
            tokenizer,
            args,
            file_path=file_path,
            block_size=args.block_size,
            start_tokens=tokenizer.ids_with_sep,
            dropout=0 if evaluate else args.bpe_dropout,
        )

        if not evaluate:
            sampler = RandomSampler(dataset) if args.local_rank == -1 else DistributedSampler(dataset)
        else:
            sampler = SequentialSampler(dataset)
        return DataLoader(
            dataset,
            sampler=sampler,
            batch_size=args.train_batch_size if not evaluate else args.eval_batch_size,
            collate_fn=collate,
            drop_last=True,
        )

    else:
        assert args.model_type == "txl"
        return LMIterator(
            file_path,
            tokenizer,
            batch_size=args.train_batch_size if not evaluate else args.eval_batch_size,
            batch_len=args.block_size,
            start_tokens=tuple(tokenizer.ids_with_string(args.example_symbol)),
            dropout_prob=0 if evaluate else args.bpe_dropout,
            overwrite_cache=args.overwrite_cache,
            shuffle=not evaluate,
            batch_first=True,
        )
