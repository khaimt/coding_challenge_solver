# Leetcode challenges solver
This repo is about building LLMs to solve code challenges from [LeetCode](https://leetcode.com/)

## Trained models

| Size | Hugging Face Repo | Base Model |
| ---  | --- | --- |
| 7B | [khaimaitien/leetcode_solver_7b](https://huggingface.co/khaimaitien/qa-expert-7B-V1.0) | [codellama/CodeLlama-7b-Instruct-hf](https://huggingface.co/codellama/CodeLlama-7b-Instruct-hf) |

## Training Data
You can download the training data from: [khaimaitien/leetcode_problem_solution](https://huggingface.co/datasets/khaimaitien/leetcode_problem_solution)

This dataset was crawled from:  https://github.com/AnasImloul/Leetcode-Solutions/tree/main

## Inference
First you need to install the requirements:
```
pip install -r requirements.txt
```
you can use this script to run inference:
```
python run_inference.py test_cases/problem1.txt --model-path khaimaitien/leetcode_solver_7b
```

Where first parameter is the leetcode problem stored in a text file, you can take a look at the format in **test_cases/problem1.txt** 

Here is the output from above command:
```python 
class Solution:
    def isUgly(self, n: int) -> bool:
        if n == 1:
            return True
        if n == 0:
            return False
        if n % 2 == 0:
            return self.isUgly(n // 2)
        if n % 3 == 0:
            return self.isUgly(n // 3)
        if n % 5 == 0:
            return self.isUgly(n // 5)
        return False
```</s>
```