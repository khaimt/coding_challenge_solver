import torch
import random
import bitsandbytes as bnb
from peft import LoraConfig
from typing import Union, List
from transformers import BitsAndBytesConfig
from training.arguments import TokenizerArguments
from transformers import PreTrainedTokenizerBase, AutoTokenizer, PreTrainedModel


def set_seed(seed: int = 100) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def config_bnb() -> BitsAndBytesConfig:
    """
    Configure bitsandbytes for QLora training
    Returns:
            BitsAndBytesConfig
    """
    # set to float16 if device is not support bfloat16
    compute_datatype = getattr(torch, "bfloat16")
    return BitsAndBytesConfig(load_in_4bit=True,
                              bnb_4bit_quant_type="nf4",
                              bnb_4bit_use_double_quant=True,
                              bnb_4bit_compute_dtype=compute_datatype)


def find_all_linear_names(model):
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, bnb.nn.Linear4bit) or isinstance(module, torch.nn.Linear):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if "lm_head" in lora_module_names:  # needed for 16-bit
        lora_module_names.remove("lm_head")
    return list(lora_module_names)


def config_peft(modules: List) -> LoraConfig:
    """
    Configure PEFT for QLora training
    Args:
        modules: define modules to apply Lora

    Returns:
        LoraConfig

    """
    return LoraConfig(
        r=16,
        lora_alpha=64,
        target_modules=modules,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
        modules_to_save=["lm_head", "embed_tokens"],
    )


def load_tokenizer(tokenizer_args: TokenizerArguments) -> PreTrainedTokenizerBase:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_args.model_id)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = tokenizer_args.padding_side

    if tokenizer_args.added_tokens:
        tokenizer.add_special_tokens({"additional_special_tokens": tokenizer_args.added_tokens})

    return tokenizer


def load_model(model_name: str) -> PreTrainedModel:
    pass
