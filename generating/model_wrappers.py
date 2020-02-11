import math
from typing import List, Dict

import torch
import torch.nn.functional as F
import tqdm
from transformers import GPT2LMHeadModel, TransfoXLLMHeadModel, TransfoXLConfig

from generating.beam_search.model_wrapper import ModelWrapper
from generating.tokenizer_wrapper import TokenizerWrapperBase, GPT2TokenizerWrapper
from mem_transformer import MemTransformerLM
from util import unwrap_model


class ModelWrapperBase(ModelWrapper):
    def __init__(self, model: torch.nn.Module, tokenizer: TokenizerWrapperBase, device: torch.device):
        self._model = ModelWrapperBase._prepare_model(model, device)
        print(f"Number of parameters: {sum(p.numel() for p in self._model.parameters())}")
        self.tokenizer = tokenizer

        self._device = device
        self._last_prefix_len = None

        self._mems = self._init_mems()

    def init_state(self, context: str, terminal_strings: List[str]) -> torch.Tensor:
        context, prefix_mask = self._tokenize_context(context, terminal_strings)

        with torch.no_grad():
            log_probs = self._init_log_probs(context)

        return self._mask_log_probs(log_probs, prefix_mask)

    def sort_state(self, sort_mask: torch.Tensor) -> None:
        with torch.no_grad():
            self._mems = [t[:, sort_mask] for t in self._mems]

    def get_log_probs(self, data: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self._get_log_probs(data)

    def reset_state(self) -> None:
        raise NotImplementedError

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.vocab_size

    @property
    def last_prefix_len(self) -> int:
        assert self._last_prefix_len is not None, f"You should use last_prefix_len only once after init_state"
        ans = self._last_prefix_len
        self._last_prefix_len = None
        return ans

    @staticmethod
    def _prepare_model(model: torch.nn.Module, device: torch.device) -> torch.nn.Module:
        model = model.to(device)
        if "cuda" in str(device):
            model = model.half()
        else:
            model = model.float()
        model = model.eval()
        return model

    def _init_mems(self):
        raise NotImplementedError

    def _tokenize_context(self, context: str, terminal_strings: List[str]):
        context, prefix_mask, self._last_prefix_len = self.tokenizer.encode_context(context, terminal_strings)
        context = torch.tensor(context).to(self._device)
        return context, prefix_mask

    def _init_log_probs(self, context: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _get_log_probs(self, data: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _mask_log_probs(self, log_probs: torch.Tensor, prefix_mask: List[int]) -> torch.Tensor:
        with torch.no_grad():
            assert tuple(log_probs.size()) == (
                1,
                self.vocab_size,
            ), f"Log probs have shape: {log_probs.size()}, not (1, vocab_size)"

            mask = torch.ones(self.vocab_size, dtype=torch.uint8)
            mask[prefix_mask] = 0
            log_probs[0, mask] = -math.inf

        return log_probs


class GPT2ModelWrapper(ModelWrapperBase):
    def __init__(self, context_len: int, device: torch.device, size: str = "medium"):
        sizes = {"distil": "distilgpt2", "small": "gpt2-small", "medium": "gpt2-medium", "large": "gpt2-large"}
        model = GPT2LMHeadModel.from_pretrained(sizes[size])

        tokenizer = GPT2TokenizerWrapper(add_special_tokens=True)

        super().__init__(model, tokenizer, device)

        self._context_len = min(context_len, 1024)

    def _init_mems(self):
        return None

    def _init_log_probs(self, context: torch.Tensor) -> torch.Tensor:
        context = context[-self._context_len :].unsqueeze(0)

        scores, self._mems = self._model(context)
        return F.log_softmax(scores[:, -1, :])

    def _get_log_probs(self, data: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            if self._mems[0].size(3) == self._context_len:
                self._mems = [t[:, :, :, 1:] for t in self._mems]

            scores, self._mems = self._model(data.unsqueeze(1), self._mems)
            log_probs = F.log_softmax(scores.squeeze(1))

            return log_probs

    def reset_state(self) -> None:
        self._mems = None


class TransfoXLWrapper(ModelWrapperBase):
    def __init__(self, config: TransfoXLConfig, device: torch.device):
        tokenizer = GPT2TokenizerWrapper()
        model = TransfoXLLMHeadModel(config)

        super().__init__(model, tokenizer, device)

        self._batch_len = 384

        self._mems = self._model.init_mems(1)

    def _init_mems(self):
        return self._model.init_mems(1)

    def _init_log_probs(self, context: torch.Tensor) -> torch.Tensor:
        context.unsqueeze_(0)

        context_batches = torch.split(context, self._batch_len, dim=1)
        assert context_batches, f"Context hasn't examples, context shape: {context.size()}"

        for batch in tqdm.tqdm(context_batches):
            scores, self._mems = self._model(batch, self._mems)

        return F.log_softmax(scores[:, -1, :])

    def _get_log_probs(self, data: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            scores, self._mems = self._model(data.unsqueeze(1), self._mems)
            log_probs = F.log_softmax(scores.squeeze(1))
            return log_probs

    def reset_state(self) -> None:
        self._mems = self._model.init_mems(1)


class MemTransformerWrapper(ModelWrapperBase):
    def __init__(
        self,
        model_path: str,
        tokenizer: TokenizerWrapperBase,
        device: torch.device,
        memory_len: int = 384,
        verbose: bool = False,
        model_params: Dict = None,
    ):
        if model_params:
            model = MemTransformerLM(n_token=tokenizer.vocab_size, **model_params)
        else:
            model = self._load_model(model_path, device)

        super().__init__(model, tokenizer, device)

        self._model.reset_length(1, 0, memory_len)
        self._batch_len = memory_len
        self._verbose = verbose

    def _init_mems(self):
        return self._model.init_mems()

    def _init_log_probs(self, context: torch.Tensor) -> torch.Tensor:
        context.unsqueeze_(1)

        context_batches = torch.split(context, self._batch_len, dim=0)
        assert context_batches, f"Context hasn't examples, context shape: {context.size()}"

        for batch in tqdm.tqdm(context_batches, disable=not self._verbose):
            hiddens, self._mems = self._predict(batch, self._mems)

        return self._hidden_to_softmax(hiddens[-1], log=True)

    def _get_log_probs(self, data: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            hiddens, self._mems = self._predict(data.unsqueeze(0), self._mems)
            log_probs = self._hidden_to_softmax(hiddens.squeeze(0), log=True)
            return log_probs

    def reset_state(self) -> None:
        self._mems = self._model.init_mems()

    def _predict(self, data, mems):
        tgt_len = data.size(0)
        hidden, new_mems = self._model._forward(data, mems=mems)
        pred_hid = hidden[-tgt_len:]
        return pred_hid, new_mems

    def _hidden_to_softmax(self, hidden, temperature=None, log=False):
        """Turn a hidden projection into softmax or log softmax.

        Adapted from utils/proj_adaptive_softmax.py
        """
        # pas stands for ProjectedAdaptiveSoftmax
        pas = self._model.crit
        logits = pas._compute_logit(hidden, pas.out_layers[0].weight, pas.out_layers[0].bias, pas.out_projs[0])

        if temperature:
            logits /= temperature
        if log:
            return F.log_softmax(logits, dim=-1)
        else:
            return F.softmax(logits, dim=-1)

    @staticmethod
    def _load_model(model_path: str, device: torch.device) -> MemTransformerLM:
        with open(model_path, "rb") as f:
            model: MemTransformerLM = torch.load(f, map_location=device)
        model = unwrap_model(model)
        return model
