import math
import os
from typing import Dict, List, Tuple

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import SequentialSampler, DataLoader
from tqdm import tqdm
from transformers import PreTrainedModel

from data_preprocessing.bpe import GitBPE
from hf_training.data_utils import load_and_cache_examples
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


def evaluate(args, model: PreTrainedModel, tokenizer: GitBPE, prefix="") -> Dict:
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_output_dir = args.output_dir

    eval_dataset = load_and_cache_examples(args, tokenizer, evaluate=True)

    if args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir, exist_ok=True)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

    # Note that DistributedSampler samples randomly

    def collate(examples: List[torch.Tensor]):
        return pad_sequence(examples, batch_first=True)

    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset,
        sampler=eval_sampler,
        batch_size=args.eval_batch_size,
        collate_fn=collate,
        num_workers=1,
        drop_last=True,
    )

    # # multi-gpu evaluate
    # if args.n_gpu > 1:
    #     model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss, acc_top1, acc_top5, MRR_top5 = 0.0, 0.0, 0.0, 0.0
    nb_eval_steps = 0
    model.eval()

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        inputs, labels = (batch, batch)
        inputs = inputs.to(args.device)
        labels = labels.to(args.device)

        with torch.no_grad():
            lm_loss, scores, *_ = model(inputs, labels=labels)

        # Loss
        eval_loss += lm_loss.mean().item()

        # Metrics
        metrics = accuracy_MRR(scores, labels, top_k=5, shift=True)
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
