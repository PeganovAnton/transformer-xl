"""Generate samples from a model.

Note: only works for BPE-based models.
Based on https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/examples/run_gpt2.py
"""
import argparse
from typing import List

import torch
import tqdm
from transformers import GPT2Tokenizer

from mem_transformer import MemTransformerLM
from prepare_git_data import prepare_project
from search import predict, hidden_to_softmax, perform_search
from util import unwrap_model


def main():
    parser = argparse.ArgumentParser(description="PyTorch Transformer Language Model")
    parser.add_argument("--model_path", type=str, required=True, help="path to the model weights checkpoint")
    parser.add_argument("--dataset", type=str, required=True, help="The dataset on which the model was trained")
    parser.add_argument("--batch_len", type=int, default=384)
    # wiki arguments
    parser.add_argument("--top_k", type=int, default=0, help="Limit sampling to top K probabilities. If 0, use all.")
    parser.add_argument("--top_p", type=float, default=0, help="Limit sampling to p nucleus sampling. If 0, use all.")
    parser.add_argument("--length", type=int, default=200, help="what sequence length to generate")
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--context", type=str, default="", help="Conditional generation context")
    # git arguments
    parser.add_argument("--num_iterations", type=int, default=50)
    parser.add_argument("--beam_size", type=int, default=5)
    parser.add_argument("--num_hyps", type=int, default=3, help="How many hypotheses from each group take. -1 for all.")
    parser.add_argument("--diversity_groups", type=int, default=5)
    parser.add_argument("--diversity_strength", type=float, default=0.3)
    parser.add_argument(
        "--cur_file", type=str, default=None, help="Only for git dataset. Conditional generation context"
    )
    parser.add_argument(
        "--context_files", type=str, default=None, nargs="+", help="Only for git dataset. Rest project files"
    )

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(args.model_path, "rb") as f:
        model = torch.load(f, map_location=device)
    model = unwrap_model(model)

    if not torch.cuda.is_available():
        model = model.float()
    model = model.to(device)
    model.eval()

    if args.dataset == "wiki":
        context = prepare_wiki_context(args.context)
        generated_text = generate_text(
            model, context, args.length, args.batch_size, args.temperature, args.top_k, args.top_p, args.batch_len
        )
    elif args.dataset == "git":
        context = prepare_git_context(args.cur_file, args.context_files)
        generated_text = generate_text(
            model,
            context,
            num_iterations=args.num_iterations,
            batch_len=args.batch_len,
            beam_size=args.beam_size,
            num_diversity_groups=args.diversity_groups,
            diversity_strength=args.diversity_strength,
        )
        generated_text = sum([group[: args.num_hyps] for group in generated_text], [])
    else:
        assert False, f"function for context preparation not implemented for dataset {args.dataset}"

    for i in range(len(generated_text)):
        print("=" * 40, "sample", i + 1, "=" * 40)
        print(generated_text[i])


def prepare_git_context(current_file: str = None, project_files: List[str] = None) -> str:
    def read_file(path):
        with open(path, "rt", encoding="utf-8", errors="ignore") as f:
            return f.read()

    project = ((project_file, read_file(project_file)) for project_file in project_files) if project_files else tuple()
    current_file = (current_file, read_file(current_file)) if current_file else None

    return prepare_project(project, current_file)


def prepare_wiki_context(context: str) -> str:
    NL = "\n" * 4
    return NL + context


def generate_text(
        model: MemTransformerLM,
        context: str,
        num_iterations: int,
        batch_len: int = 384,
        beam_size: int = 5,
        num_diversity_groups: int = 5,
        diversity_strength: float = 0.3,
        tokenizer: GPT2Tokenizer = None,
        terminating_symbols: List[str] = None,
        verbose: bool = True,
) -> List[List[str]]:
    model.eval()

    if not tokenizer:
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    context = tokenizer.encode(context)

    if terminating_symbols is None:
        # Use '\n' as default
        terminating_symbols = ['\n', '(', ')', '[', ']', ':', '->', ',', '.']
    terminal_ids = get_ids_with_symbols(terminating_symbols, tokenizer)

    with torch.no_grad():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        data = torch.tensor(context).to(device)
        # Turn into a batch.
        data = data.unsqueeze(1)

        # Init mems with context
        context_batches = torch.split(data, batch_len, dim=0)
        mems = model.init_mems()
        if verbose:
            print("Prepare context for text generating...")
        for batch in tqdm.tqdm(context_batches, disable=not verbose):
            hiddens, mems = predict(model, batch, mems)

        # Get the last log_probs for the first beam search iteration
        log_probs = hidden_to_softmax(model, hiddens[-1], log=True)

        # Generate text
        if verbose:
            print("Performing beam search...")
        results = search(
            model=model,
            mems=mems,
            log_probs=log_probs,
            num_iterations=num_iterations,
            terminal_id=terminal_id,
            beam_size=beam_size,
            num_groups=num_diversity_groups,
            diversity_strength=diversity_strength,
        )
    return [[tokenizer.decode(hypothesis.tolist()) for hypothesis, score in group] for group in results]


def get_ids_with_symbols(symbols: List[str], tokenizer: GPT2Tokenizer) -> List[int]:
    def convert_token_to_string(token):
        # Code this because old versions of HF's tokenizers don't support it
        return bytearray([tokenizer.byte_decoder[c] for c in token]).decode('utf-8', errors="ignore")

    result = []
    for token, token_id in tokenizer.encoder.items():
        if any(symbol in convert_token_to_string(token) for symbol in symbols):
            result.append(token_id)
    return result


if __name__ == "__main__":
    main()
