import json
from typing import List, Dict
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from prompt_template.base_template import PromptTemplate


class AlgoDataset(Dataset):
    def __init__(self, data_path: str,
                 tokenizer: PreTrainedTokenizer,
                 prompt_template: PromptTemplate,
                 batch_size: int = 8,
                 max_seq_len: int = None):
        super().__init__()
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.prompt_template = prompt_template
        self.algo_data = self.load_algo_dataset()
        print("Length of algo dataset: ", len(self.algo_data))
        # self.training_data = self.convert_to_training_data()
        self.training_data = self.convert_to_prompt_data()

    def __len__(self):
        return len(self.training_data)

    def __getitem__(self, idx):
        return self.training_data[idx]

    def load_algo_dataset(self) -> List[Dict[str, str]]:
        """
        Load dataset from disk
        Returns:
            List of dictionaries with keys 'title', 'algo_input', 'solution_py'

        """
        with open(self.data_path, 'r') as f:
            algo_data = json.load(f)

            return algo_data

    def convert_to_training_data(self) -> List[Dict]:
        """
        Convert raw data to list of dictionaries prepare for training
        Returns:
            A list of dictionaries: [{"input_ids": ..., "attention_mask": ..., "labels": ...}]

        """
        print("Converting raw data to list of dictionaries...")
        training_data = []
        for idx in range(0, len(self.algo_data), 1):
            batch_inputs = self.algo_data[idx: idx + 1]
            examples = self.prompt_template.prepare_training_inputs(batch_inputs=batch_inputs,
                                                                    tokenizer=self.tokenizer,
                                                                    padding="longest",
                                                                    max_length=self.max_seq_len,
                                                                    return_tensor=True)
            print("examples: ", examples)
            training_data.extend(examples)

        return training_data

    def convert_to_prompt_data(self) -> List[Dict]:
        training_data = []
        for sample in self.algo_data:
            prompt = (
                f"[INST] Write code to solve the following coding problem that obeys"
                f"the constraints and passes the example test cases."
                f"Please wrap your code answer using ```:\n{sample['algo_input']}\n[/INST]```python\n"
            )
            output = sample['solution_py']

            prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False, return_tensors=None)
            output_ids = self.tokenizer.encode(output, add_special_tokens=False, return_tensors=None)
            input_ids = prompt_ids + output_ids
            labels = len(prompt_ids) * [-100] + output_ids

            training_data.append({
                "input_ids": input_ids,
                "labels": labels,
                "length": len(input_ids)
            })
        training_data = sorted(training_data, key=lambda _sample: _sample["length"], reverse=True)
        return training_data


if __name__ == '__main__':
    def collate_fn(batch):
        print(batch)


    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer, LlamaTokenizerFast
    from prompt_template.code_llama_template import CodellamaTemplate

    _tokenizer = AutoTokenizer.from_pretrained('codellama/CodeLlama-7b-Instruct-hf', model_max_length=4096)
    _tokenizer.pad_token = _tokenizer.eos_token
    _tokenizer.padding_side = "right"

    print(_tokenizer.model_max_length)
    codellama_template = CodellamaTemplate()
    algo_dataset = AlgoDataset(data_path="../data/algorithm_train.json",
                               batch_size=4,
                               tokenizer=_tokenizer,
                               prompt_template=codellama_template)

    dataloader = DataLoader(dataset=algo_dataset,
                            batch_size=4,
                            collate_fn=collate_fn,
                            shuffle=False)

    for batch in dataloader:
        print(batch['labels'])

    from transformers import AutoTokenizer, DataCollatorWithPadding
