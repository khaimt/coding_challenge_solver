import torch
import typer
import requests
from typing import Union
from bs4 import BeautifulSoup
from prompt_template import get_prompt_template_by_name
from transformers import AutoModelForCausalLM, AutoTokenizer


def crawl_desc_leetcode_url(url: str) -> Union[str, None]:
    """Crawl algorithm description from leetcode"""
    url = url.strip()
    if "leetcode.com" not in url:
        raise ValueError(
            "Please input url from leetcode. "
            "For example: 'https://leetcode.com/problems/regular-expression-matching/")
    response = requests.get(url)
    response.raise_for_status()
    response.encoding = "utf-8"
    html = response.text
    if html:
        soup = BeautifulSoup(html, 'html.parser')

        # Find the meta tag with name="description"
        meta_description = soup.find('meta', attrs={'name': 'description'})

        # Extract the content attribute value
        if meta_description:
            content_value = meta_description.get('content')
            return content_value
        else:
            return None
    else:
        return None


def main(leetcode_url: str = None, input_file: str = None, model_path: str = "khaimaitien/leetcode_solver_7b"):
    """
    Run the inference from url or input file
    Args:
        leetcode_url: url from leetcode
        input_file: leetcode problem
        model_path: path to save the model

    Returns:

    """
    print("model_path: ", model_path)
    if leetcode_url is None and input_file is None:
        typer.echo("Please provide leetcode URL or input file!", err=True, color=True)
        return None
    prompt_template = get_prompt_template_by_name("code_llama")
    algo_input = None
    if input_file is not None:
        # read the leetcode problem from input_file
        with open(input_file, "r") as f:
            algo_input = f.read()

    if leetcode_url is not None:
        algo_input = crawl_desc_leetcode_url(leetcode_url)

    if algo_input is None:
        typer.echo("Could not read problem!", err=True, color=True)
        return None

    prompt_str = prompt_template.get_prompt_from_inputs({"algo_input": algo_input})

    typer.echo(50 * "=" + "PROMPT" + 50 * "=" + "\n")
    typer.echo(prompt_str)
    typer.echo(100 * "=" + "\n")

    # Load tokenizer and model
    typer.echo(f"Loading model from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.bfloat16)
    typer.echo("Model loaded!")

    token_ids = tokenizer([prompt_str], return_tensors="pt")["input_ids"]
    token_ids = token_ids.to(model.device)
    outputs = model.generate(input_ids=token_ids, max_new_tokens=1024, do_sample=True, temperature=0.0001)
    all_token_ids = outputs[0].tolist()
    output_token_ids = all_token_ids[token_ids.shape[-1]:]
    output = tokenizer.decode(output_token_ids)
    print("\n\n\n==============================Solution generated from Model==============================\n")

    print(output.replace("</s>", "").replace("```", ""))


if __name__ == "__main__":

    typer.run(main)
