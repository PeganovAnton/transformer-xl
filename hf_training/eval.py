import math
import os
from typing import Dict, Tuple

import torch
from tqdm import tqdm
from transformers import PreTrainedModel

from hf_training.log import logger


def accuracy_MRR(
    scores: torch.Tensor, labels: torch.Tensor, top_k: int = 5, shift: bool = False
) -> Tuple[float, float, float]:
    assert scores.ndimension() == labels.ndimension() + 1
    assert scores.size()[:-1] == labels.size()
    assert scores.size(-1) >= top_k

    if shift:
        scores = scores[:, :-1]
        labels = labels[:, 1:]

    # Top predictions
    _, top5 = torch.topk(scores, top_k)
    true_pos = top5 == labels.unsqueeze(-1).expand_as(top5)

    # Accuracy top 1
    acc_top1 = float(true_pos[:, :, 0].sum().item()) / labels.nelement()
    # Accuracy top 5
    acc_topk = float(true_pos[:, :, :5].sum().item()) / labels.nelement()

    # MRR top
    MRR_topk = (
        true_pos.to(dtype=torch.double)
        / (torch.arange(end=true_pos.size(-1), dtype=torch.double, device=true_pos.device) + 1)
    ).sum().item() / labels.nelement()

    return acc_top1, acc_topk, MRR_topk


def evaluate(args, model: PreTrainedModel, eval_data_iterator, prefix="") -> Dict:
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_output_dir = args.output_dir

    if args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir, exist_ok=True)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_data_iterator))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss, acc_top1, acc_top5, MRR_top5 = 0.0, 0.0, 0.0, 0.0
    nb_eval_steps = 0
    mems = tuple()
    model.eval()

    for batch in tqdm(eval_data_iterator, desc="Evaluating"):
        if args.model_type == "gpt-2":
            inputs, labels = (batch, batch)
        else:
            assert args.model_type == "txl"
            inputs, labels = batch

        inputs = inputs.to(args.device)
        labels = labels.to(args.device)

        with torch.no_grad():
            if args.model_type == "gpt-2":
                lm_loss, scores, *_ = model(inputs, labels=labels)
            else:
                assert args.model_type == "txl"
                lm_loss, scores, *mems = model(*mems, input_ids=inputs, labels=labels, return_scores=True)

        # Loss
        eval_loss += lm_loss.mean().item()

        # Metrics
        if args.model_type == "gpt-2":
            metrics = accuracy_MRR(scores, labels, top_k=5, shift=True)
        else:
            assert args.model_type == "txl"
            metrics = accuracy_MRR(scores, labels, top_k=5, shift=False)
        acc_top1 += metrics[0]
        acc_top5 += metrics[1]
        MRR_top5 += metrics[2]

        nb_eval_steps += 1

    eval_loss, acc_top1, acc_top5, MRR_top5 = (
        eval_loss / nb_eval_steps,
        acc_top1 / nb_eval_steps,
        acc_top5 / nb_eval_steps,
        MRR_top5 / nb_eval_steps,
    )

    result = {
        "loss": eval_loss,
        "perplexity": math.exp(eval_loss),
        "accuracy_top1": acc_top1,
        "accuracy_top5": acc_top5,
        "MRR_top5": MRR_top5,
    }

    output_eval_file = os.path.join(eval_output_dir, prefix, "eval_results.txt")
    with open(output_eval_file, "w") as writer:
        logger.info("***** Eval results {} *****".format(prefix))
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))

    return result
