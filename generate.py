"""Generate samples from a model.

Note: only works for BPE-based models.
Based on https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/examples/run_gpt2.py
"""
import argparse
from typing import List

import torch
import tqdm

from beam_search import beam_search, predict
from mem_transformer import MemTransformerLM
from prepare_git_data import prepare_project
from util import unwrap_model


def main():
    parser = argparse.ArgumentParser(description="PyTorch Transformer Language Model")
    parser.add_argument("--model_path", type=str, required=True, help="path to the model weights checkpoint")
    parser.add_argument("--dataset", type=str, required=True, help="The dataset on which the model was trained")
    parser.add_argument("--top_k", type=int, default=0, help="Limit sampling to top K probabilities. If 0, use all.")
    parser.add_argument("--top_p", type=float, default=0, help="Limit sampling to p nucleus sampling. If 0, use all.")
    parser.add_argument("--length", type=int, default=200, help="what sequence length to generate")
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--batch_len", type=int, default=384)
    parser.add_argument("--temperature", type=float, default=1.0)
    # Only for wiki dataset
    parser.add_argument("--context", type=str, default="", help="Conditional generation context")
    # Only for git dataset
    parser.add_argument(
        "--cur_file", type=str, default=None, help="Only for git dataset. Conditional generation context"
    )
    parser.add_argument(
        "--context_files", type=str, default=None, nargs="+", help="Only for git dataset. Rest project files"
    )

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the best saved model.
    with open(args.model_path, "rb") as f:
        model = torch.load(f, map_location="cuda" if torch.cuda.is_available() else "cpu")
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

    generated_text = generate_text(
        model, context, args.length, args.batch_size, args.temperature, args.top_k, args.top_p, args.batch_len
    )

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
        length: int,
        num_examples: int = 5,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 0.0,
        batch_len: int = 384,
        tokenizer=None,
        verbose=True,
) -> List[str]:
    model.eval()
    if not tokenizer:
        from pytorch_pretrained_bert import GPT2Tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    context = tokenizer.encode(context)
    terminal_id = tokenizer.encode("\n")
    assert len(terminal_id) == 1
    terminal_id = terminal_id[0]

    with torch.no_grad():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        data = torch.tensor(context).to(device)
        # Turn into a batch.
        data = data.unsqueeze(1)

        # Init mems with context, except for the last token
        context_batches = torch.split(data[:-1], batch_len, dim=0)
        mems = model.init_mems()
        if verbose:
            print("Prepare context for text generating...")
        for batch in tqdm.tqdm(context_batches, disable=not verbose):
            _, mems = predict(model, batch, mems)

        # Generate text
        if verbose:
            print("Performing beam search...")
        results = beam_search(model=model, terminal_id=terminal_id, beam_size=num_examples, mems=mems, data=data)
    #     for _ in tqdm.trange(length, disable=not verbose):
    #         # Grab a sample from the last frame, append to result list, append to `data`
    #         pred_hid, mems = predict(model, data[-1:], mems)
    #         softmax = hidden_to_softmax(model, pred_hid.squeeze(0), temperature=temperature)
    #         new_sample = torch.multinomial(softmax, num_samples=1)
    #         data = torch.cat((data, new_sample.t()), dim=0)
    #
    # results = []
    # for i in range(data.size(1)):
    #     results.append(tokenizer.decode(data[len(context):, i].tolist()))

    return [tokenizer.decode(result.tolist()) for result in results]


if __name__ == "__main__":
    main()
