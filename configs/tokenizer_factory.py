from configs.config_types import TokenizerWrapperConfig, GPT2TokenizerConfig, GitBPETokenizerConfig
from generating.tokenizer_wrapper import TokenizerWrapperBase, GPT2TokenizerWrapper, GitBPETokenizerWrapper


def get_tokenizer_wrapper(config: TokenizerWrapperConfig) -> TokenizerWrapperBase:
    if isinstance(config, GPT2TokenizerConfig):
        tokenizer_wrapper = GPT2TokenizerWrapper(**config.as_dict())

    elif isinstance(config, GitBPETokenizerConfig):
        tokenizer_wrapper = GitBPETokenizerWrapper(**config.as_dict())

    else:
        raise TypeError(f"config must be an TokenizerWrapperConfig instance, not {type(config)}")

    return tokenizer_wrapper
