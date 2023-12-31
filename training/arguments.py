from transformers import TrainingArguments
from typing import List, Union, Optional
from dataclasses import dataclass, field


@dataclass
class TokenizerArguments:
    _model_name_or_path: Optional[str] = field(default="codellama/CodeLlama-7b-Instruct-hf",
                                               metadata={"help": "Tokenizer model name on HuggingFace"})

    padding_side: Optional[str] = field(default="left",
                                        metadata={"help": "Setting padding side is left or right"})

    added_tokens: Optional[Union[List[str], None]] = field(default=None,
                                                           metadata={"help": "Define tokens added to model"})


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="codellama/CodeLlama-7b-Instruct-hf",
                                              metadata={"help": "mode name or path to pretrained model"})
    model_type: str = field(default="llama")
    use_lora: bool = field(default=True)
    model_max_length: int = field(
        default=4096,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    qlora: bool = field(default=True, metadata={"help": "whether using qlora or not"})


@dataclass
class DataArguments:
    train_path: str = field(default="", metadata={"help": "Path to the training data."})
    validation_path: str = field(default="", metadata={"help": "Path to the evaluation data"})


@dataclass
class TrainingArguments(TrainingArguments):
    per_device_train_batch_size: int = field(default=8,
                                             metadata={
                                                 "help": "The batch size per GPU/XPU/TPU/MPS/NPU core/CPU for training"})
    per_device_eval_batch_size: int = field(default=8,
                                            metadata={
                                                "help": "The batch size per GPU/XPU/TPU/MPS/NPU core/CPU for evaluation"})
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    packing: bool = field(default=False, metadata={"help": "Whether use packing or not"})
