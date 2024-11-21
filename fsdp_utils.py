"""
Script adopted from:
https://github.com/facebookresearch/llama-recipes/blob/main/src/llama_recipes/policies/wrapping.py
"""
import functools

import torch
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers.models.mistral.modeling_mistral import MistralDecoderLayer


def get_llama_wrapper():
    """we register our main layer class and use the fsdp transformer wrapping policy
    ensures embedding layers are in the root fsdp unit for shared access and that fsdp units map to transformer layers
    """
    llama_auto_wrap_policy = functools.partial(
        torch.distributed.fsdp.wrap.transformer_auto_wrap_policy,
        transformer_layer_cls={
            LlamaDecoderLayer,
        },
    )

    return llama_auto_wrap_policy


def get_mistral_wrapper():
    """we register our main layer class and use the fsdp transformer wrapping policy
    ensures embedding layers are in the root fsdp unit for shared access and that fsdp units map to transformer layers
    """
    mistral_auto_wrap_policy = functools.partial(
        torch.distributed.fsdp.wrap.transformer_auto_wrap_policy,
        transformer_layer_cls={
            MistralDecoderLayer,
        },
    )

    return mistral_auto_wrap_policy
