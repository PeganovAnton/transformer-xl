from typing import List

import numpy as np
import torch
from torch.nn import functional as F

from mem_transformer import MemTransformerLM


def beam_search(
        *, model: MemTransformerLM, terminal_id: int, beam_size: int, mems: List[torch.Tensor], data: torch.Tensor
) -> List[torch.Tensor]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    assert (
            data.ndimension() == 2
    ), f"Data must have two dimensions (#tokens, batch_size), actual dimensions are {data.size()}"
    assert data.size(1) == 1, f"Batch size must be 1, but {data.size(1)} was given"
    for mem in mems:
        assert mem.size(1) == 1, f"You must provide mems only for 1 example, but {mem.size(1)} was given"

    # Index of hypothesis start in the data tensor
    hyp_ind = data.size(0)

    # First dimension for hypotheses
    # (#tokens, 1) -> (1, #tokens)
    data = data.squeeze(-1).unsqueeze(0)
    # (mem_len, 1, hidden_size) -> (1, mem_len, hidden_size)
    mems = [mem.transpose(0, 1) for mem in mems]

    lengths = torch.zeros(1, dtype=torch.int16, device=device)
    is_terminated_mask = torch.zeros(1, dtype=torch.uint8, device=device)
    log_probs = torch.zeros(1, dtype=torch.float16, device=device)

    while True:
        # data[:, -1]: (batch_size,) -> (1, batch_size)
        # mems: (batch_size, mem_len, hidden_size) -> (mem_len, batch_size, hidden_size)
        predicted_hiddens, mems = predict(model, data[:, -1].unsqueeze(0), [mem.transpose(0, 1) for mem in mems])

        # (mem_len, batch_size, hidden_size) -> (batch_size, mem_len, hidden_size)
        mems = [mem.transpose(0, 1) for mem in mems]

        log_softmax = hidden_to_softmax(model, predicted_hiddens.squeeze(0), log=True)

        (data, lengths, is_terminated_mask, log_probs), new_log_probs, new_samples = _expand_hypotheses(
            data, lengths, is_terminated_mask, log_probs, softmax=log_softmax, beam_size=beam_size
        )
        mems = _expand_hypotheses(*mems, beam_size=beam_size)

        # Update log_probs
        log_probs += new_log_probs

        # Update is_terminated_mask
        is_terminated_mask |= new_samples == terminal_id

        # Update lengths
        lengths += (1 - is_terminated_mask).to(torch.int16)

        # Update data
        data = torch.cat((data, new_samples.unsqueeze(1)), dim=1)

        scores = _score_hypotheses(log_probs, lengths, data[:, hyp_ind:], is_terminated_mask, alpha=0.8)

        data, lengths, is_terminated_mask, log_probs = _get_best_hypotheses(
            data, lengths, is_terminated_mask, log_probs, scores=scores, beam_size=beam_size
        )
        mems = _get_best_hypotheses(*mems, scores=scores, beam_size=beam_size)

        if is_terminated_mask.all():
            break

    return [hyp[hyp_ind: hyp_ind + length] for hyp, length in zip(data, lengths)]


def _score_hypotheses(
        log_probs: torch.Tensor,
        lengths: torch.Tensor,
        hypotheses: torch.Tensor,
        is_terminated_mask: torch.Tensor,
        alpha: float = 0,
) -> torch.Tensor:
    assert log_probs.ndimension() == 1, ""
    assert lengths.ndimension() == 1, ""
    assert hypotheses.ndimension() == 2, ""
    assert is_terminated_mask.ndimension() == 1, ""

    # TODO: Add diversity reward calculation
    reward = 0

    scores = log_probs.to(torch.float64) / (lengths.to(torch.float64) + 1e-6) + alpha * reward
    return scores


def _get_best_hypotheses(*tensors: torch.Tensor, scores: torch.Tensor, beam_size: int):
    _, top_mask = torch.topk(scores, beam_size)
    return tuple(tensor[top_mask] for tensor in tensors)


def _expand_hypotheses(*tensors: torch.Tensor, beam_size: int, softmax: torch.Tensor = None):
    repeat_index = torch.tensor(np.repeat(np.arange(tensors[0].size(0)), beam_size))
    repeated_tensors = tuple(tensor[repeat_index] for tensor in tensors)
    if softmax is not None:
        probs, samples = torch.topk(softmax, beam_size, sorted=False)
        # (batch_size, beam_size) -> (batch_size * beam_size,)
        probs, samples = probs.flatten(), samples.flatten()
        return repeated_tensors, probs, samples
    return repeated_tensors


def predict(model, data, mems):
    tgt_len = data.size(0)
    with torch.no_grad():
        hidden, new_mems = model._forward(data, mems=mems)
    pred_hid = hidden[-tgt_len:]
    return pred_hid, new_mems


def hidden_to_softmax(model, hidden, temperature=1.0, top_k=0, top_p=0.0, log=False):
    """Turn a hidden projection into softmax or log softmax.

    Adapted from utils/proj_adaptive_softmax.py
    """
    # pas stands for ProjectedAdaptiveSoftmax
    pas = model.crit
    logits = pas._compute_logit(hidden, pas.out_layers[0].weight, pas.out_layers[0].bias, pas.out_projs[0])
    logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)

    logits /= temperature
    if log:
        return F.log_softmax(logits, dim=-1)
    else:
        return F.softmax(logits, dim=-1)


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float("Inf")):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering

    https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317

        Args:
            logits: logits distribution shape (..., vocabulary size)
            top_k >0: keep only top k tokens with highest probability (top-k filtering).
            top_p >0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
    """
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs >= top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = torch.zeros_like(logits, dtype=torch.uint8).scatter_(
            dim=-1, index=sorted_indices, src=sorted_indices_to_remove
        )
        logits[indices_to_remove] = filter_value
    return logits
