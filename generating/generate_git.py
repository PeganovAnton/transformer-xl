"""Generate samples from a model trained on git dataset."""
import argparse
from typing import List, Tuple

from dataclasses import asdict

from configs import get_config_by_name
from configs.configs import (
    SequenceGeneratingConfig,
    GPT2ModelWrapperConfig,
    TransformerXLWrapperConfig,
    TransformerXLHFWrapperConfig,
)
from generating.model_wrappers import MemTransformerWrapper, GPT2ModelWrapper, TransfoXLWrapper
from generating.beam_search.sequence_generator import SequenceGenerator
from prepare_git_data import GitDataPreprocessor


def generate_sequences(config: SequenceGeneratingConfig) -> List[List[Tuple[str, float]]]:
    data_preprocessor = GitDataPreprocessor(**asdict(config.git_data_preprocessor_config))

    if isinstance(config.model_config, TransformerXLWrapperConfig):
        model_wrapper = MemTransformerWrapper(**asdict(config.model_config))

    elif isinstance(config.model_config, GPT2ModelWrapperConfig):
        model_wrapper = GPT2ModelWrapper(**asdict(config.model_config))

    elif isinstance(config.model_config, TransformerXLHFWrapperConfig):
        model_wrapper = TransfoXLWrapper(**asdict(config.model_config))

    else:
        raise NameError

    sequence_generator = SequenceGenerator(
        model_wrapper=model_wrapper,
        terminal_ids=model_wrapper.tokenizer.find_ids_with_strings(config.beam_search_config.terminal_strings),
        beam_size=config.beam_search_config.beam_size,
        num_groups=config.beam_search_config.num_groups,
        diversity_strength=config.beam_search_config.diversity_strength,
        verbose=config.beam_search_config.verbose,
    )

    with open(args.cur_file, "rt") as f:
        content = f.read()
    context = data_preprocessor.prepare_context(tuple(), (args.cur_file, content))
    print(f"\nContext:\n{context}\n")

    sequences = sequence_generator.search_sequence(
        num_iterations=config.num_iterations,
        context=context,
        terminal_strings=config.beam_search_config.terminal_strings,
    )

    prefix_len = model_wrapper.last_prefix_len

    decoded_groups = []
    for group in sequences:
        decoded_strings = model_wrapper.tokenizer.decode([seq for seq, score in group], prefix_len)
        decoded_groups.append(list(zip(decoded_strings, ([score for seq, score in group]))))

    return decoded_groups


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GitHub dataset sequence generation")
    parser.add_argument("--config", type=str, required=True, help="configs name")
    parser.add_argument("--cur_file", type=str, default=None, help="Conditional generation context")
    parser.add_argument("--context_files", type=str, default=None, nargs="+", help="Rest project files")

    args = parser.parse_args()

    config: SequenceGeneratingConfig = get_config_by_name(args.config)
    assert isinstance(
        config, SequenceGeneratingConfig
    ), f"You need pass SequenceGeneratingConfig as configs, not {type(config)}"

    groups = generate_sequences(config)

    for i, group in enumerate(groups):
        print(f"---------------group #{i + 1}---------------")
        for seq, score in group[:5]:
            print(f"{round(score, 2)}: {seq}")
