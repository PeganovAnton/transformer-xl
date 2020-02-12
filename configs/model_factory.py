from configs.config_types import (
    ModelWrapperConfig,
    TransformerXLWrapperConfig,
    GPT2ModelWrapperConfig,
    TransformerXLHFWrapperConfig,
)
from generating.model_wrappers import ModelWrapperBase, MemTransformerWrapper, GPT2ModelWrapper, TransfoXLWrapper


def get_model_wrapper(config: ModelWrapperConfig) -> ModelWrapperBase:
    if isinstance(config, TransformerXLWrapperConfig):
        model_wrapper = MemTransformerWrapper(**config.as_dict())

    elif isinstance(config, GPT2ModelWrapperConfig):
        model_wrapper = GPT2ModelWrapper(**config.as_dict())

    elif isinstance(config, TransformerXLHFWrapperConfig):
        model_wrapper = TransfoXLWrapper(**config.as_dict())

    else:
        raise NameError

    return model_wrapper
