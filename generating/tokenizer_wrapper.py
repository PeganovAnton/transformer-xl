from typing import List, Tuple, Dict

from transformers import GPT2Tokenizer

from generating.trie import TrieNode


class TokenizerWrapperBase:
    """Tokenizer wrapper with functions that needed for sequence generation"""

    def __init__(self):
        self._trie = self._build_trie()

    def encode_context(self, context: str, ignoring_strings_in_prefix: List[str]) -> Tuple[List[int], List[int], int]:
        """Encodes context.
        TODO:

        :param context: string with context
        :returns ids: ids of encoded context except the last token
        :returns prefix ids: ids of tokens that start with the last token
        """
        assert context, f"Context {context} is empty"
        ids = self._tokenize(context)
        assert ids, f"There are no ids ({ids}) after tokenization {context}"

        prefix = self._id_to_token(ids[-1])
        prefix_mask = self._find_ids_with_prefix(prefix)

        if any(terminal in prefix for terminal in ignoring_strings_in_prefix) or len(prefix_mask) == 1:
            return ids, self._find_ids_with_prefix(""), 0
        return ids[:-1], prefix_mask, len(prefix)

    def decode(self, sequences: List[List[int]], prefix_len) -> List[str]:
        results = self._decode(sequences)
        return [res[prefix_len:] for res in results]

    @property
    def vocab_size(self) -> int:
        raise NotImplementedError

    def _decode(self, sequences: List[List[int]]) -> List[str]:
        raise NotImplementedError

    @property
    def _vocab(self) -> Dict[str, int]:
        raise NotImplementedError

    def _tokenize(self, text: str) -> List[int]:
        raise NotImplementedError

    def _id_to_token(self, index: int) -> str:
        raise NotImplementedError

    def find_ids_with_strings(self, strings: List[str]) -> List[int]:
        result = []
        for word, index in self._vocab.items():
            if any(string in word for string in strings):
                result.append(index)
        return result

    def _find_ids_with_prefix(self, prefix: str) -> List[int]:
        answer = self._trie.get_values(prefix)
        assert answer, f"Prefix {prefix} isn't a prefix to any token in vocabulary"
        return answer

    def _build_trie(self) -> TrieNode:
        root = TrieNode()
        for word, value in self._vocab.items():
            root.add_word(word, value)
        return root


class GPT2TokenizerWrapper(TokenizerWrapperBase):
    def __init__(self, add_special_tokens: bool = True):
        self._add_special_tokens = add_special_tokens
        self._tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self._vocabulary = {
            self._tokenizer.convert_tokens_to_string([token]): index for token, index in self._tokenizer.encoder.items()
        }
        super().__init__()

    @property
    def vocab_size(self) -> int:
        return self._tokenizer.vocab_size

    def _decode(self, sequences: List[List[int]]) -> List[str]:
        return [
            self._tokenizer.decode(sequence, skip_special_tokens=self._add_special_tokens) for sequence in sequences
        ]

    @property
    def _vocab(self) -> Dict[str, int]:
        return self._vocabulary

    def _tokenize(self, text: str) -> List[int]:
        return self._tokenizer.encode(text, add_special_tokens=self._add_special_tokens)

    def _id_to_token(self, index: int) -> str:
        return self._tokenizer.convert_tokens_to_string(self._tokenizer.convert_ids_to_tokens([index]))
