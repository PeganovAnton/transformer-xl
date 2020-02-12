"""Config is a frozen dataclass.
That allows accessing attributes as field names (like AttrDict) and using type hints.
You can find method replace from dataclasses module useful while working with configs."""
import re

from dataclasses import replace

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
    min_content_len=50,
    project_level=False,
    old_style=False,
)

git_data_preprocessor_project = replace(git_data_preprocessor_file_level, project_level=True)

git_data_preprocessor_old = replace(git_data_preprocessor_file_level, old_style=True)

# Tokenizer configs
gpt2_tokenizer_txl = GPT2TokenizerConfig(add_special_tokens=False)
gpt2_tokenizer_hf = GPT2TokenizerConfig(add_special_tokens=True)

git_bpe_line_5000 = GitBPETokenizerConfig(path_to_tokenizer="search-Line-5000.bpe")

# Model evaluation configs
transformerxl_wrapper_gpt2tok_old = TransformerXLWrapperConfig(
    model_path="git360-90.5-model.pt",
    memory_len=384,
    tokenizer_config=gpt2_tokenizer_txl,
    device=device,
    verbose=True,
    model_params=None,
)

# Some configs for benchmarking
tokenizer_config = git_bpe_line_5000
vocab_size = 5000

# tokenizer_config = gpt2_tokenizer_hf
# vocab_size = 50257

gpt2_wrapper_medium = GPT2ModelWrapperConfig(
    tokenizer_config=tokenizer_config,
    device=device,
    config=GPT2Config(vocab_size=vocab_size, n_embd=1024, n_layer=24, n_head=16),
    context_len=384,
    from_pretrained_name=None,
)

transfoxl_HF = TransformerXLHFWrapperConfig(
    tokenizer_config=tokenizer_config,
    device=device,
    config=TransfoXLConfig(
        vocab_size=vocab_size, div_val=1, n_layer=24, d_inner=3072, mem_len=384, tgt_len=384, adaptive=False, cutoffs=[]
    ),
    mem_len=120,
    from_pretrained_name=None,
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
    beam_search_config=beam_search_config_line,
)

sequence_generation_gpt2_medium = replace(sequence_generation_txl_gpt2tok_old, model_config=gpt2_wrapper_medium)

sequence_generation_transfoxl = replace(sequence_generation_txl_gpt2tok_old, model_config=transfoxl_HF)
