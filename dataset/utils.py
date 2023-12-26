from __future__ import annotations
from typing import List, Any, Optional


# This function is copied from:
# https://github.com/MeetKai/functionary/blob/main/functionary/train/custom_datasets.py#L61
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


# This function is copied from:
# https://github.com/MeetKai/functionary/blob/main/functionary/train/custom_datasets.py#L212
def get_masked_labels(
        *,
        input_token_ids: List[int],
        tokenizer: Any,
        assistant_prefix_tokens: List[List[int]],
        assistant_stop_tokens: List[int],
        keep_assistant_prefix: bool = False,
        verbose: bool = False) -> List:
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
                chunk_ids = input_token_ids[start: end_index + 1] if end_index > -1 else input_token_ids[start:]
                print("chunk_ids: ", chunk_ids)
                print(
                    "longer chunk: ",
                    input_token_ids[index: end_index + 1] if end_index > 1 else input_token_ids[index:],
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
