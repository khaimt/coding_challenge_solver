import torch
import argparse
from peft import PeftModel
from transformers import BitsAndBytesConfig
from transformers import AutoTokenizer, LlamaForCausalLM


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-path_adapter',
                        help='Path to adapter folder.', required=True)
    args = parser.parse_args()

    # Model
    model_name = "codellama/CodeLlama-7b-Instruct-hf"
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    compute_datatype = getattr(torch, "bfloat16")

    model = LlamaForCausalLM.from_pretrained(model_name,
                                             device_map={"": 0},
                                             quantization_config=BitsAndBytesConfig(load_in_4bit=True,
                                                                                    bnb_4bit_quant_type="nf4",
                                                                                    bnb_4bit_use_double_quant=True,
                                                                                    bnb_4bit_compute_dtype=compute_datatype)
                                             )

    # Configure the pad token in the model
    model.config.pad_token_id = tokenizer.pad_token_id
    model = PeftModel.from_pretrained(model, args.path_adapter)

    def generate(instruction):
        inputs = tokenizer(instruction, return_tensors="pt", add_special_tokens=False)
        input_ids = inputs["input_ids"]
        input_ids = input_ids.to(model.device)
        generation_output = model.generate(
            input_ids=input_ids,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=1024,
            top_k=50,
            num_beams=5,
            do_sample=False

        )
        for seq in generation_output.sequences:
            output = tokenizer.decode(seq)
            return output.split("[/INST]")[-1]

    while True:
        fp = input("Enter path to problem file: ")
        fp = fp.strip()
        if fp.lower() == 'q':
            break
        else:
            with open(fp, "r") as f:
                problem = f.read()

                f.close()

            prompt_str = (
                f"[INST] Write code to solve the following coding problem that obeys"
                f"the constraints and passes the example test cases."
                f"Please wrap your code answer using ```:\n{problem.strip()}\n[/INST]```python\n"
            )
            print(f"Problem:")
            print("=" * 20)
            print(f"{problem.strip()}")
            print("=" * 20)

            solution = generate(prompt_str)
            print("Solution:")
            print("=" * 20)

            print(solution)
            print()
            print()


if __name__ == '__main__':
    main()
