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

### There are 2 options to run inference

#### Option 1: Run inference from input file.

```bash
python run_inference.py --input-file test_cases/problem1.txt --model-path khaimaitien/leetcode_solver_7b
```
_Where first parameter is the leetcode problem stored in a text file, you can take a look at the format in **test_cases/problem1.txt**_ 
    
Here is the output from above command:
``` python
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
```


#### Option 2: Run inference from leetcode URL.
  
```bash
 python run_inference.py --leetcode-url https://leetcode.com/problems/merge-k-sorted-lists/ --model-path khaimaitien/leetcode_solver_7b
```

Here is the output from above command:
``` python
class Solution:
    def mergeKLists(self, lists: List[ListNode]) -> ListNode:
        if not lists:
            return None
        return self.merge(lists)
    
    def merge(self, lists):
        if len(lists) == 1:
            return lists[0]
        mid = len(lists) // 2
        l1 = self.merge(lists[:mid])
        l2 = self.merge(lists[mid:])
        return self.mergeTwoLists(l1, l2)
    
    def mergeTwoLists(self, l1, l2):
        if not l1:
            return l2
        if not l2:
            return l1
        if l1.val < l2.val:
            l1.next = self.mergeTwoLists(l1.next, l2)
            return l1
        else:
            l2.next = self.mergeTwoLists(l1, l2.next)
            return l2
```