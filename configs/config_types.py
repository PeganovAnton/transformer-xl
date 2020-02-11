from typing import Dict, List, Union

import torch
from dataclasses import dataclass
from transformers import TransfoXLConfig

from generating.tokenizer_wrapper import TokenizerWrapperBase


# Preprocessing config types
@dataclass(frozen=True)
class GitDataPreprocessingConfig:
    min_content_len: int
    example_split_symbol: str
    file_split_symbol: str
    filepath_split_symbol: str
    project_level: bool
    old_style: bool = False


# Model config types
@dataclass(frozen=True)
class ModelWrapperConfig:
    device: torch.device


@dataclass(frozen=True)
class TransformerXLWrapperConfig(ModelWrapperConfig):
    model_path: str
    memory_len: int
    tokenizer: TokenizerWrapperBase
    verbose: bool
    model_params: Union[None, Dict]


@dataclass(frozen=True)
class GPT2ModelWrapperConfig(ModelWrapperConfig):
    context_len: int
    size: str


@dataclass(frozen=True)
class TransformerXLHFWrapperConfig(ModelWrapperConfig):
    config: TransfoXLConfig


# Tokenizer configs
@dataclass(frozen=True)
class GPT2TokenizerConfig:
    add_special_tokens: bool


# Beam search config types
@dataclass(frozen=True)
class BeamSearchConfig:
    terminal_strings: List[str]
    beam_size: int
    num_groups: int = 1
    diversity_strength: Union[float, None] = 0.3
    verbose: bool = True


# Sequence generator config types
@dataclass(frozen=True)
class SequenceGeneratingConfig:
    num_iterations: int
    git_data_preprocessor_config: GitDataPreprocessingConfig
    model_config: ModelWrapperConfig
    tokenizer_config: GPT2TokenizerConfig
    beam_search_config: BeamSearchConfig
