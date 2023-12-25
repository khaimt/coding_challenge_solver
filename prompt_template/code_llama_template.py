from prompt_template.base_template import PromptTemplate
from typing import List, Dict, Any


class CodellamaTemplate(PromptTemplate):
    @property
    def name(self) -> str:
        return "code_llama"

    def get_stop_tokens_for_generation(self) -> List[str]:
        return ["</s>"]

    def get_prompt_from_inputs(self, inputs: Dict[str, Any]) -> str:
        algo_input = inputs["algo_input"]
        prompt_str = (
            f"[INST] Write code to solve the following coding problem that obeys"
            f"the constraints and passes the example test cases."
            f"Please wrap your code answer using ```:\n{algo_input}\n[/INST]"
        )
        algo_solution = inputs.get("solution_py", None)
        if algo_solution:
            prompt_str += f"{algo_solution}</s>"
        return prompt_str

    def get_response_prefixs(self) -> List[str]:
        """This function returns a list of prefixs that are included into prompt before generating the response
        For example, "\nResponse:", "<end>\nAssistant:"
        Returns:
            List[str]: _description_
        """
        return ["\n[/INST]"]
