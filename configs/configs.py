"""Config is a frozen dataclass.
That allows accessing attributes as field names (like AttrDict) and using type hints."""

from dataclasses import replace, asdict
from transformers import GPT2Tokenizer

from configs.config_types import *


def get_config_by_name(name: str):
    assert re.match("\\w+", name), f"Config name should be a one word, but {name} was given"
    assert name in globals(), f"There is no configs called {name}"
    config = eval(name)
    return config


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Here go configs

# Data preprocessing configs
git_data_preprocessor_file_level = GitDataPreprocessingConfig(
    min_content_len=50, example_split_symbol="␢", file_split_symbol="ℱ", filepath_split_symbol="₣", project_level=False
)

git_data_preprocessor_project = replace(git_data_preprocessor_file_level, project_level=True)

git_data_preprocessor_old = replace(git_data_preprocessor_file_level, old_style=True)

# Tokenizer configs
gpt2_tokenizer_txl = GPT2TokenizerConfig(add_special_tokens=False)
gpt2_tokenizer_hf = GPT2TokenizerConfig(add_special_tokens=True)

# Model evaluation configs
gpt2_wrapper_medium = GPT2ModelWrapperConfig(context_len=384, device=device, size="medium")

transformerxl_wrapper_gpt2tok_old = TransformerXLWrapperConfig(
    model_path="git360-90.5-model.pt",
    memory_len=384,
    tokenizer=GPT2TokenizerWrapper(**asdict(gpt2_tokenizer_txl)),
    device=device,
    verbose=True,
    model_params=None,
)

transformerxl_gpt2med_eq = replace(
    transformerxl_wrapper_gpt2tok_old,
    model_params={
        "n_layer": 24,
        "d_model": 1024,
        "n_head": 16,
        "d_head": 64,
        "d_inner": 3072,
        "dropout": 0.2,
        "dropatt": 0.2,
        "tgt_len": 384,
        "mem_len": 384,
        "ext_len": 0,
    },
)

transfoxl_HF = TransformerXLHFWrapperConfig(
    config=TransfoXLConfig(
        vocab_size=GPT2Tokenizer.from_pretrained("gpt2").vocab_size,
        div_val=1,
        n_layer=24,
        d_inner=3072,
        mem_len=384,
        tgt_len=384,
    ),
    device=device,
)

# Beam search configs
beam_search_config_line = BeamSearchConfig(
    terminal_strings=["\n"], beam_size=6, num_groups=1, diversity_strength=None, verbose=True
)

# Sequence generation configs
sequence_generation_txl_gpt2tok_old = SequenceGeneratingConfig(
    num_iterations=50,
    git_data_preprocessor_config=git_data_preprocessor_old,
    model_config=transformerxl_wrapper_gpt2tok_old,
    tokenizer_config=gpt2_tokenizer_txl,
    beam_search_config=beam_search_config_line,
)

sequence_generation_gpt2_medium = replace(
    sequence_generation_txl_gpt2tok_old, model_config=gpt2_wrapper_medium, tokenizer_config=gpt2_tokenizer_hf
)

sequence_generation_transfoxl = replace(sequence_generation_txl_gpt2tok_old, model_config=transfoxl_HF)

sequence_generation_txl_gpt2med_eq = replace(sequence_generation_txl_gpt2tok_old, model_config=transformerxl_gpt2med_eq)
