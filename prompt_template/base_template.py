from __future__ import annotations
from abc import abstractmethod
from typing import List, Dict, Any, Optional
import torch


# This function is copied from: https://github.com/MeetKai/functionary/blob/main/functionary/train/custom_datasets.py#L61
def get_matching_prefix(prefix_tokens: List[List[int]], sequence_ids: List[int]) -> Optional[List[int]]:
    """This function is used to check if sequence_ids starts with any prefix

    Args:
        prefix_tokens (List[List[int]]): _description_
        sequence_ids (List[int]): _description_

    Returns:
        List[int]: _description_
    """
    for prefix in prefix_tokens:
        if len(sequence_ids) >= len(prefix):
            if sequence_ids[: len(prefix)] == prefix:
                return prefix
    return None


# This function is copied from: https://github.com/MeetKai/functionary/blob/main/functionary/train/custom_datasets.py#L212
def get_masked_labels(
    *,
    input_token_ids: List[int],
    tokenizer: Any,
    assistant_prefix_tokens: List[List[int]],
    assistant_stop_tokens: List[int],
    keep_assistant_prefix: bool = False,
    verbose: bool = False,
):
    """This function is used to mask labels.
    This will retain only chunks: (prefix assistant tokens) CHUNK_TO_UNMASK (stop tokens) for computing loss

    Args:
        input_token_ids (List[int]): input_token_ids
        tokenizer (Any): _description_
        assistant_prefix_tokens (List[List[int]]): _description_
        assistant_stop_tokens (List[int]): _description_
        keep_assistant_prefix (bool, optional): _description_. Defaults to False.
        verbose (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    # first we initialize labels with all positions as -100,
    # then we will fill in positions where role=assistant as we only include these in computing the loss
    labels = [-100 for _ in range(len(input_token_ids))]
    start = 0
    # now we will unmask labels by positions that was from assistant
    # we will find the chunks: "<endtoken>assistant ...(<end_of_function>|<end_of_assistant>) from input_token_ids
    # and unmask: this part: "...(<end_of_function>|<end_of_assistant>"
    # find token_ids of: "<endtoken>assistant"
    # prefix_token_ids = get_prefix_assistant_token_ids(tokenizer)
    # if verbose:
    #    print("prefix_token_ids: ", prefix_token_ids)
    index = 0
    total_input_leng = len(input_token_ids)
    while index < total_input_leng:
        # finding the index that start with: "<endtoken>assistant" --> we will unmask labels from this position
        matched_prefix = get_matching_prefix(assistant_prefix_tokens, input_token_ids[index:])
        if matched_prefix is not None:
            end_index = -1
            # unmask until reach <end_of_function> or <end_of_assistant>
            start_masked_index = index + len(matched_prefix)
            if keep_assistant_prefix:  # unmask prefix of assistant
                start_masked_index = index

            for i in range(start_masked_index, total_input_leng):
                tok_id = input_token_ids[i]
                if tok_id in assistant_stop_tokens:  # check if this is end of turn
                    labels[i] = input_token_ids[i]  # unmask labels at this position
                    end_index = i
                    break
                else:
                    labels[i] = input_token_ids[i]  # unmask labels at this position

            if verbose:
                print("------------------------")
                start = start_masked_index  # index + len(matched_prefix)
                chunk_ids = input_token_ids[start : end_index + 1] if end_index > -1 else input_token_ids[start:]
                print("chunk_ids: ", chunk_ids)
                print(
                    "longer chunk: ",
                    input_token_ids[index : end_index + 1] if end_index > 1 else input_token_ids[index:],
                )
                print(f"chunk:{tokenizer.decode(chunk_ids)}")
                print("-------------------")
            if (
                end_index == -1
            ):  # if at the end, cannot find EndToken.assistant or EndToken.function_call --> this data point was truncated
                break
            index = end_index
        else:
            index += 1
    return labels


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
    def get_response_prefixs(self) -> List[str]:
        """This function returns a list of prefixs that are included into prompt before generating the response
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

    # This function is based on this: https://github.com/MeetKai/functionary/blob/main/functionary/train/custom_datasets.py#L318
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
        response_prefixs = self.get_response_prefixs()
        # first get the token_ids of response prefixs
        response_prefix_token_ids = []
        for prefix in response_prefixs:
            response_prefix_token_ids.append(self.get_token_ids(prefix, tokenizer))

        # assistant_stop_token_ids
        stop_token_ids = []
        for token in self.get_stop_tokens_for_generation():
            token_id = tokenizer.encode(token, add_special_tokens=False)[-1]
            stop_token_ids.append(token_id)

        prompt_str_list = []
        for inputs in batch_inputs:
            prompt_str_list.append(self.get_prompt_from_inputs(inputs))
        max_length = max_length if max_length is not None else tokenizer.model_max_length

        input_dic = tokenizer(prompt_str_list, padding=padding, max_length=max_length, truncation=True)

        batch_labels = []
        for input_token_ids in input_dic["input_ids"]:
            labels = get_masked_labels(
                input_token_ids=input_token_ids,
                tokenizer=tokenizer,
                assistant_prefix_tokens=response_prefix_token_ids,
                assistant_stop_tokens=stop_token_ids,
                keep_assistant_prefix=False,
                verbose=False,
            )

        batch_labels.append(labels)
        assert len(labels) == len(input_token_ids)
        input_dic["labels"] = batch_labels
        assert (
            len(input_dic["labels"])
            == len(input_dic["input_ids"])
            == len(input_dic["attention_mask"])
            == len(batch_inputs)
        )

        batch_inputs = []
        for i in range(len(input_dic["input_ids"])):
            inputs = {}
            for key in ["labels", "input_ids", "attention_mask"]:
                inputs[key] = input_dic[key][i]
                if return_tensor:
                    inputs[key] = torch.tensor(inputs[key])
            batch_inputs.append(inputs)

        return batch_inputs

    @classmethod
    def get_prompt_template(cls) -> PromptTemplate:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
