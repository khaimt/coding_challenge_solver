from __future__ import annotations
import torch
from abc import abstractmethod
from typing import List, Dict, Any, Optional

from dataset import utils


class PromptTemplate:
    _instance = None

    @property
    def name(self) -> str:
        """Return the name of the template

        Raises:
            NotImplementedError: _description_

        Returns:
            str: _description_
        """
        raise NotImplementedError

    @abstractmethod
    def get_stop_tokens_for_generation(self) -> List[str]:
        """Return a list of stop tokens to mark the end of generated text

        Raises:
            NotImplementedError: _description_

        Returns:
            List[str]: _description_
        """
        raise NotImplementedError

    @abstractmethod
    def get_prompt_from_inputs(self, inputs: Dict[str, Any]) -> str:
        """This function is used to convert inputs --> string. This will be used for training and inference
        inputs is a dictionary str --> Any
        Args:
            inputs (Dict): _description_

        Raises:
            NotImplemented: _description_

        Returns:
            str: the prompt created from inputs
        """
        raise NotImplementedError

    @abstractmethod
    def get_response_prefixes(self) -> List[str]:
        """This function returns a list of prefixes that are included into prompt before generating the response
        For example, "\nResponse:", "<end>\nAssistant:"
        Returns:
            List[str]: _description_
        """
        raise NotImplementedError

    def get_removed_token_ids_in_tokenizer_encode(self) -> List[int]:
        """sometimes, tokenizer.encode(text, add_special_tokens=False) still add additional tokens at the start of the result.
        For example, in Llama2: 29871; in Mistral: 28705
        We will remove this when use tokenizer.encode

        Returns:
            List[int]: _description_
        """
        return [29871, 28705]

    def get_token_ids(self, text: str, tokenizer: Any) -> List[int]:
        """Tokenize a text into token_ids: List[int]

        Args:
            text (str): _description_
            tokenizer (Any): _description_

        Returns:
            List[int]: _description_
        """
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        if token_ids[0] in self.get_removed_token_ids_in_tokenizer_encode():
            token_ids = token_ids[1:]
        return token_ids

    def get_additional_tokens(self) -> List[str]:
        """This function return a list of added tokens for this template

        Returns:
            List[str]: List of strings, each is an added token
        """
        return []

    # This function is based on this:
    # https://github.com/MeetKai/functionary/blob/main/functionary/train/custom_datasets.py#L318
    def prepare_training_inputs(
            self,
            batch_inputs: List[Dict[str, Any]],
            tokenizer: Any,
            padding: Optional[str] = "max_length",
            max_length: Optional[int] = None,
            return_tensor: bool = True,
    ) -> List[Dict]:
        """This function is used to prepare inputs for training

        Args:
            batch_inputs (List[Dict[str, Any]]): List of inputs, each inputs is a dictionary
            tokenizer (Any): Tokenizer
            padding (Optional[str], optional): padding used in tokenizer. Defaults to "max_length".
            max_length (Optional[int], optional): max_length for tokenizer. Defaults to None.
            return_tensor (bool, optional): return a tensor or int. Defaults to True.

        Returns:
            List[Dict]: A list of dictionaries: [{"input_ids": ..., "attention_mask": ..., "labels": ...}, ...]
        """
        response_prefixes = self.get_response_prefixes()
        # first get the token_ids of response response_prefixes
        response_prefix_token_ids = []
        for prefix in response_prefixes:
            response_prefix_token_ids.append(self.get_token_ids(prefix, tokenizer))

        # assistant_stop_token_ids
        stop_token_ids = []
        for token in self.get_stop_tokens_for_generation():
            token_id = tokenizer.encode(token, add_special_tokens=False)[-1]
            stop_token_ids.append(token_id)

        prompt_str_list = []
        for inputs in batch_inputs:
            _prompt_str = self.get_prompt_from_inputs(inputs)
            if _prompt_str:
                prompt_str_list.append(_prompt_str)

        # Config max_length for model
        if max_length is None:
            max_length = tokenizer.model_max_length
        else:
            max_length = min(max_length, tokenizer.model_max_length)
        input_dict = tokenizer(prompt_str_list,
                               padding=padding,
                               max_length=max_length,
                               truncation=True)
        labels = []
        input_token_ids = []
        batch_labels = []
        for input_token_ids in input_dict["input_ids"]:
            labels = utils.get_masked_labels(
                input_token_ids=input_token_ids,
                tokenizer=tokenizer,
                assistant_prefix_tokens=response_prefix_token_ids,
                assistant_stop_tokens=stop_token_ids,
                keep_assistant_prefix=False,
                verbose=False,
            )

            batch_labels.append(labels)
        assert len(labels) == len(input_token_ids)
        input_dict["labels"] = batch_labels
        assert (
                len(input_dict["labels"])
                == len(input_dict["input_ids"])
                == len(input_dict["attention_mask"])
                == len(batch_inputs)
        )

        batch_inputs = []
        for i in range(len(input_dict["input_ids"])):
            inputs = {}
            for key in ["labels", "input_ids", "attention_mask"]:
                inputs[key] = input_dict[key][i]
                if return_tensor:
                    inputs[key] = torch.tensor(inputs[key])
            batch_inputs.append(inputs)

        return batch_inputs

    @classmethod
    def get_prompt_template(cls) -> PromptTemplate:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
