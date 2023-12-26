from typing import List, Dict, Any, Union
from prompt_template.base_template import PromptTemplate


class CodellamaTemplate(PromptTemplate):
    @property
    def name(self) -> str:
        return "code_llama"

    def get_stop_tokens_for_generation(self) -> List[str]:
        return ["</s>"]

    def get_prompt_from_inputs(self, inputs: Dict[str, Any]) -> Union[str, None]:
        """
        Build a PromptTemplate for each algorithm
        Args:
            inputs: {"algo_input": str, "solution_py": str}

        Returns:

        """
        algo_input = inputs.get("algo_input")
        if algo_input is None:
            return None
        prompt_str = (
            f"[INST] Write code to solve the following coding problem that obeys"
            f"the constraints and passes the example test cases."
            f"Please wrap your code answer using ```:\n{algo_input}\n[/INST]```python\n"
        )
        algo_solution = inputs.get("solution_py")
        if algo_solution is None:
            return None

        prompt_str += f"{algo_solution.strip()}\n```</s>"
        return prompt_str

    def get_response_prefixes(self) -> List[str]:
        """This function returns a list of prefixes that are included into prompt before generating the response
        For example, "\nResponse:", "<end>\nAssistant:"
        Returns:
            List[str]: _description_
        """
        return ["\n[/INST]```python\n"]
