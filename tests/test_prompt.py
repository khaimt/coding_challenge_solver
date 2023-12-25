from prompt_template import get_prompt_template_by_name, get_all_template_names, PromptTemplate
import unittest
import os
import json
from transformers import AutoTokenizer


class PrompTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(PrompTest, self).__init__(*args, **kwargs)
        current_folder = os.path.dirname(os.path.abspath(__file__))
        self.test_case_folder = os.path.join(current_folder, "test_cases")
        test_case_path = os.path.join(self.test_case_folder, "case.json")
        with open(test_case_path, "r") as f:
            self.test_case = json.loads(f.read())

    def test_all_prompt_templates(self):
        for name in get_all_template_names():
            print(f"TEST template: {name}")
            self.run_test_with_template(name)

    def run_test_with_template(self, template_name: str):
        template = get_prompt_template_by_name(template_name)
        prompt_str = template.get_prompt_from_inputs(self.test_case)

        correct_prompt_path = os.path.join(self.test_case_folder, f"prompts/{template.name}.txt")
        with open(correct_prompt_path, "r") as f:
            correct_prompt = f.read()

        # first make sure that the template is correct
        self.assertEqual(prompt_str.strip(), correct_prompt.strip(), "prompt is diffrent")

        # next, make sure that masking labels is correct
        tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-7b-Instruct-hf", legacy=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.add_special_tokens({"additional_special_tokens": template.get_additional_tokens()})

        batch_inputs = template.prepare_training_inputs(
            [self.test_case], tokenizer, padding="max_length", max_length=1024
        )
        labels = batch_inputs[0]["labels"]

        # remove text not included in computing loss
        unmasked_token_ids = [label for label in labels if label != -100]
        unmasked_text = tokenizer.decode(unmasked_token_ids)
        # First make sure that unmasked_text endswith stop_token
        end_with_stop_token = False
        for token in template.get_stop_tokens_for_generation():
            if unmasked_text.endswith(token):
                unmasked_text = unmasked_text[: -len(token)].strip()
                end_with_stop_token = True
                break
        self.assertTrue(end_with_stop_token)
        # make sure that unmasked_text == self.test_case["solution_py"]
        self.assertEqual(unmasked_text, self.test_case["solution_py"])


if __name__ == "__main__":
    unittest.main()
