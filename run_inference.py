from transformers import AutoModelForCausalLM, AutoTokenizer
from prompt_template import get_prompt_template_by_name
import torch
import typer
import requests
from bs4 import BeautifulSoup


def crawl_desc_leetcode_url(url: str) -> str:
    """Crawl algorithm description from leetcode"""
    url = url.strip()
    if "leetcode.com" not in url:
        raise ValueError(
            f"Please input url to leetcode problem. "
            f"For example: 'https://leetcode.com/problems/regular-expression-matching/")
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


def main(url: str, model_path: str = typer.Option("khaimaitien/leetcode_solver_7b")):
    prompt_template = get_prompt_template_by_name("code_llama")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.bfloat16)

    # read the leetcode problem from input_file
    # with open(input_file, "r") as f:
    #     algo_input = f.read()

    algo_input = crawl_desc_leetcode_url(url)
    if algo_input is None:
        print("Can not crawl problem from leetcode.")
        return None

    prompt_str = prompt_template.get_prompt_from_inputs({"algo_input": algo_input})
    print("--------prompt-------")
    print(prompt_str)
    token_ids = tokenizer([prompt_str], return_tensors="pt")["input_ids"]
    token_ids = token_ids.to(model.device)
    outputs = model.generate(input_ids=token_ids, max_new_tokens=1024, do_sample=True, temperature=0.0001)
    all_token_ids = outputs[0].tolist()
    ouput_token_ids = all_token_ids[token_ids.shape[-1] :]
    output = tokenizer.decode(ouput_token_ids)
    print("-------------Solution generated from Model---------")
    print(output)


if __name__ == "__main__":
    # typer.run(main)
    reader = crawl_desc_leetcode_url("https://aaa.com/problems/substring-with-concatenation-of-all-words/")
    print(reader)
