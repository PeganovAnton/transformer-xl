"""Generate samples from a model.

Note: only works for BPE-based models.
Based on https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/examples/run_gpt2.py
"""
import argparse
from typing import List

import torch
import torch.nn.functional as F
import tqdm

from mem_transformer import MemTransformerLM
from util import unwrap_model


def main():
    parser = argparse.ArgumentParser(description='PyTorch Transformer Language Model')
    parser.add_argument('--model_path', type=str, required=True,
                        help='path to the model weights checkpoint')
    parser.add_argument('--dataset', type=str, required=True,
                        help='The dataset on which the model was trained')
    parser.add_argument('--top_k', type=int, default=0,
                        help='Limit sampling to top K probabilities. If 0, use all.')
    parser.add_argument('--top_p', type=float, default=0,
                        help='Limit sampling to p nucleus sampling. If 0, use all.')
    parser.add_argument('--length', type=int, default=200,
                        help='what sequence length to generate')
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--batch_len', type=int, default=384)
    parser.add_argument("--temperature", type=float, default=1.0)
    # Only for wiki dataset
    parser.add_argument('--context', type=str, default='',
                        help='Conditional generation context')
    # Only for git dataset
    parser.add_argument('--cur_file', type=str, default=None,
                        help='Only for git dataset. Conditional generation context')
    parser.add_argument('--context_files', type=str, default=None, nargs='+',
                        help='Only for git dataset. Rest project files')


    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the best saved model.
    with open(args.model_path, 'rb') as f:
        model = torch.load(f, map_location='cuda' if torch.cuda.is_available() else 'cpu')
    model = unwrap_model(model)

    if not torch.cuda.is_available():
        model = model.float()
    model = model.to(device)
    model.eval()

    if args.dataset == "wiki":
        context = prepare_wiki_context(args.context)
    elif args.dataset == "git":
        context = prepare_git_context(args.cur_file, args.context_files)
    else:
        assert False, f"function for context preparation not implemented for dataset {args.dataset}"

    generated_text = generate_text(model, context, args.length, args.batch_size,
                                   args.temperature, args.top_k, args.top_p, args.batch_len)

    for i in range(len(generated_text)):
        print('=' * 40, 'sample', i + 1, '=' * 40)
        print(generated_text[i])


def prepare_git_context(context_file: str = None, project_files: List[str] = None) -> str:
    from prepare_git_data import EXAMPLE_SPLIT_SYMBOL as FILE_SYMBOL
    context = FILE_SYMBOL
    if context_file:
        context += f"<<!<<{context_file}>>!>>\n{open(context_file, 'rt', encoding='utf-8', errors='ignore').read()}"
    return context


def prepare_wiki_context(context: str) -> str:
    NL = "\n" * 4
    return NL + context


def generate_text(model: MemTransformerLM, context: str, length: int,
                  num_examples: int = 1, temperature: float = 1.0, top_k: int = 0, top_p: float = 0.,
                  batch_len: int = 384, tokenizer=None, verbose=True) -> List[str]:
    model.eval()
    if not tokenizer:
        from pytorch_pretrained_bert import GPT2Tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    context = tokenizer.encode(context)

    with torch.no_grad():
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        data = torch.tensor(context).to(device)
        # Turn into a batch.
        data = data.unsqueeze(1).repeat_interleave(num_examples, dim=1)

        # Init mems with context, except for the last token
        context_batches = torch.split(data[:-1], batch_len, dim=0)
        mems = model.init_mems()
        if verbose:
            print("Prepare context for text generating...")
        for batch in tqdm.tqdm(context_batches, disable=not verbose):
            _, mems = predict(model, batch, mems)

        # Generate text
        if verbose:
            print("Generate text...")
        for _ in tqdm.trange(length, disable=not verbose):
            # Grab a sample from the last frame, append to result list, append to `data`
            pred_hid, mems = predict(model, data[-1:], mems)
            softmax = hidden_to_softmax(model, pred_hid.squeeze(0), top_k=top_k, temperature=temperature, top_p=top_p)
            new_sample = torch.multinomial(softmax, num_samples=1).unsqueeze(-1).squeeze(2)
            data = torch.cat((data, new_sample.t()), dim=0)

    results = []
    for i in range(data.size(1)):
        results.append(tokenizer.decode(data[len(context):, i].tolist()))

    return results


def predict(model, data, mems):
    tgt_len = data.size(0)
    with torch.no_grad():
        hidden, new_mems = model._forward(data, mems=mems)
    pred_hid = hidden[-tgt_len:]
    return pred_hid, new_mems


def hidden_to_softmax(model, hidden, temperature=1., top_k=0, top_p=0.):
    """Turn a hidden projection into log softmax.

    Adapted from utils/proj_adaptive_softmax.py
    """
    # pas stands for ProjectedAdaptiveSoftmax
    pas = model.crit
    logits = pas._compute_logit(hidden, pas.out_layers[0].weight,
                                pas.out_layers[0].bias, pas.out_projs[0])
    logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)

    logits /= temperature
    softmax = F.softmax(logits, dim=-1)
    return softmax


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
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
            dim=-1, index=sorted_indices, src=sorted_indices_to_remove )
        logits[indices_to_remove] = filter_value
    return logits


if __name__ == '__main__':
    main()
