from typing import Dict, List, Union

import torch
from dataclasses import dataclass, fields
from transformers import TransfoXLConfig, GPT2Config


@dataclass(frozen=True)
class BaseConfig:
    def as_dict(self) -> Dict:
        """Unlike asdict method from dataclasses module, this method isn't recursive and doesn't copy attributes."""
        result = []
        for f in fields(self):
            name = f.name
            value = getattr(self, name)
            result.append((name, value))
        return dict(result)


# Preprocessing config types
@dataclass(frozen=True)
class GitDataPreprocessingConfig(BaseConfig):
    min_content_len: int
    project_level: bool
    example_split_symbol: str = "␢"
    file_split_symbol: str = "ℱ"
    filepath_split_symbol: str = "₣"
    old_style: bool = False


# Tokenizer configs
@dataclass(frozen=True)
class TokenizerWrapperConfig(BaseConfig):
    pass


@dataclass(frozen=True)
class GPT2TokenizerConfig(TokenizerWrapperConfig):
    add_special_tokens: bool


@dataclass(frozen=True)
class GitBPETokenizerConfig(TokenizerWrapperConfig):
    path_to_tokenizer: str


# Model config types
@dataclass(frozen=True)
class ModelWrapperConfig(BaseConfig):
    tokenizer_config: TokenizerWrapperConfig
    device: torch.device


@dataclass(frozen=True)
class TransformerXLWrapperConfig(ModelWrapperConfig):
    model_path: str
    memory_len: int = 384
    verbose: bool = True
    model_params: Union[Dict, None] = None


@dataclass(frozen=True)
class GPT2ModelWrapperConfig(ModelWrapperConfig):
    config: Union[GPT2Config, None]
    context_len: int
    from_pretrained_name: Union[str, None]


@dataclass(frozen=True)
class TransformerXLHFWrapperConfig(ModelWrapperConfig):
    config: Union[TransfoXLConfig, None]
    mem_len: int
    from_pretrained_name: Union[str, None]


# Beam search config types
@dataclass(frozen=True)
class BeamSearchConfig(BaseConfig):
    terminal_strings: List[str]
    beam_size: int
    num_groups: int = 1
    diversity_strength: Union[float, None] = None
    verbose: bool = True


# Sequence generator config types
@dataclass(frozen=True)
class SequenceGeneratingConfig(BaseConfig):
    num_iterations: int
    git_data_preprocessor_config: GitDataPreprocessingConfig
    model_config: ModelWrapperConfig
    beam_search_config: BeamSearchConfig
