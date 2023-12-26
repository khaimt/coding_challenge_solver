import os
import re
import json
from typing import *
from pathlib import Path


def cleanhtml(raw_html):
    clean_pattern = re.compile('<.*?>')

    cleantext = re.sub(clean_pattern, '', raw_html)
    return cleantext


def remove_redundant_line(input_text: str):
    """
    Remove redundant lines from solution.
    For example:# Runtime: 78 ms (Top 92.98%) | Memory: 14.2 MB (Top 33.69%)
    Args:
        input_text:

    Returns:

    """

    # Split the input text into lines
    lines = input_text.split('\n')

    # Remove lines containing "Runtime" or "Memory"
    filtered_lines = [line for line in lines if "Runtime" not in line and "Memory" not in line]

    # Join the filtered lines back into a single string
    result_text = '\n'.join(filtered_lines)

    return result_text.strip("\n")


def read_algo(folder_path: Union[Path, str], output_path: Union[Path, str]) -> List[Dict]:
    """
    Read all algorithm from https://github.com/AnasImloul/Leetcode-Solutions/tree/main
    Args:
        folder_path: path to algorithm folder in repo
        output_path: path to save output

    Returns: List of dict algorithms

    """
    counter = 0
    algorithm_list: List[Dict] = []
    for folder_name, sub_folders, filenames in os.walk(folder_path):
        title = Path(folder_name).stem
        algo_input = None
        solution_py = None  # solution with python programing
        solution_js = None  # solution with javascript programing
        solution_java = None  # solution with java programing
        solution_c = None  # solution with C programing
        counter += 1
        for filename in filenames:
            file_path = os.path.join(folder_name, filename)

            if os.path.isfile(file_path):
                suffix = Path(file_path).suffix
                try:
                    file_content = open(file_path, 'r').read()
                    if suffix == '.md':
                        algo_input = cleanhtml(file_content)
                        algo_lines = algo_input.split("\n")
                        first_line = algo_lines[0]
                        if title.lower() in first_line.lower():
                            algo_input = "\n".join(algo_lines[1:])
                    elif suffix == '.py':
                        solution_py = remove_redundant_line(file_content)
                    elif suffix == '.js':
                        solution_js = remove_redundant_line(file_content)
                    elif suffix == '.java':
                        solution_java = remove_redundant_line(file_content)
                    elif suffix == '.cpp':
                        solution_c = remove_redundant_line(file_content)
                except Exception as e:
                    print(file_path)
        algo = {
            "title": title,
            "algo_input": algo_input,
            "solution_py": solution_py,
            "solution_js": solution_js,
            "solution_java": solution_java,
            "solution_c": solution_c
        }
        if algo_input and solution_py and solution_js and solution_java and solution_c:
            algorithm_list.append(algo)
    print("Number of Algorithms:", len(algorithm_list))
    print("Total Number of Algorithms:", counter)
    print(algorithm_list[100])

    with open(os.path.join(output_path, "algorithm.json"), "w") as fp:
        json.dump(algorithm_list, fp, ensure_ascii=False, indent=4)
    return algorithm_list


# folder_path = './Leetcode-Solutions/scripts/algorithms/'
# read_algo(folder_path, "./")

if __name__ == '__main__':
    from sklearn.model_selection import train_test_split

    with open("algorithm.json") as fp:
        algorithm_list = json.load(fp)
        algo_train, algo_val = train_test_split(algorithm_list, test_size=0.1, random_state=100, shuffle=True)

        with open("algorithm_train.json", "w") as fp:
            json.dump(algo_train, fp, ensure_ascii=False, indent=4)
        with open("algo_val.json", "w") as fp:
            json.dump(algo_val, fp, ensure_ascii=False, indent=4)
