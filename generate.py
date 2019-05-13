"""Generate samples from a model.

Note: only works for BPE-based models.
Based on https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/examples/run_gpt2.py
"""
import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
import tqdm

from pytorch_pretrained_bert import GPT2Tokenizer


def main():
    parser = argparse.ArgumentParser(description='PyTorch Transformer Language Model')
    parser.add_argument('--work_dir', type=str, required=True,
                        help='path to the work_dir')
    parser.add_argument('--context', type=str, default='',
                        help='Conditional generation context')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    parser.add_argument('--top_k', type=int, default=0,
                        help='Limit sampling to top K probabilities. If 0, use all.')
    parser.add_argument('--length', type=int, default=200,
                        help='what sequence length to generate')
    parser.add_argument('--batch_size', type=int, default=10,
                        help='what sequence length to generate')
    parser.add_argument("--temperature", type=float, default=1.0)


    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # Load the best saved model.
    with open(os.path.join(args.work_dir, 'model-best.pt'), 'rb') as f:
        model = torch.load(f)

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    NL = tokenizer.encode('\n')

    model = model.to(device)
    model.eval()

    ## Init
    data = torch.tensor(NL*4 + tokenizer.encode(args.context)).to(device)
    # Turn into a batch.
    data.unsqueeze_(1)
    data = data.repeat_interleave(args.batch_size, dim=1)

    if not hasattr(model, 'init_mems'):
        model = model.module
    mems = model.init_mems()

    for i in tqdm.trange(args.length):
        ## Grab a sample from the last frame, append to result list, append to `data`
        # TODO: using mems breaks generation. Find a way to fix?
        pred_hid, mems_ = predict(model, data, mems)
        softmax = hidden_to_softmax(model, pred_hid[-1], top_k=args.top_k, temperature=args.temperature)

        new_sample = torch.multinomial(softmax, num_samples=1).unsqueeze(-1).squeeze(2)
        data = torch.cat((data, new_sample.t()), dim=0)

    for i in range(data.size(1)):
        print('=' * 40, 'sample', i + 1, '=' * 40)
        # Chop off the newlines before printing
        print(tokenizer.decode(data[4:, i].tolist()))

def predict(model, data, mems):
    tgt_len = data.size(0)
    with torch.no_grad():
        hidden, new_mems = model._forward(data, mems=mems)
    pred_hid = hidden[-tgt_len:]
    return pred_hid, new_mems

def hidden_to_softmax(model, hidden, temperature=1, top_k=0):
    """Turn a hidden projection into log softmax.

    Adapted from utils/proj_adaptive_softmax.py
    """
    # pas stands for ProjectedAdaptiveSoftmax
    pas = model.crit
    logits = pas._compute_logit(hidden, pas.out_layers[0].weight,
                                pas.out_layers[0].bias, pas.out_projs[0])
    logits = top_k_logits(logits, k=top_k)

    logits /= temperature
    softmax = F.softmax(logits, dim=-1)
    return softmax


def top_k_logits(logits, k):
    """
    Masks everything but the k top entries as -infinity (1e10).
    Used to mask logits such that e^-infinity -> 0 won't contribute to the
    sum of the denominator.
    """
    if k == 0:
        return logits
    else:
        values = torch.topk(logits, k)[0]
        batch_mins = values[:, -1].view(-1, 1).expand_as(logits)
        return torch.where(logits < batch_mins, torch.ones_like(logits) * -1e10, logits)


if __name__ == '__main__':
    main()
