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
        self.training_data = self.convert_to_training_data()

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
        for idx in range(0, len(self.algo_data), self.batch_size):
            print("Processing batch", idx)
            batch_inputs = self.algo_data[idx: idx + self.batch_size]
            training_data.extend(self.prompt_template.prepare_training_inputs(batch_inputs=batch_inputs,
                                                                              tokenizer=self.tokenizer,
                                                                              padding="longest",
                                                                              max_length=self.max_seq_len,
                                                                              return_tensor=True))

        return training_data


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer, LlamaTokenizerFast
    from prompt_template.code_llama_template import CodellamaTemplate

    _tokenizer = AutoTokenizer.from_pretrained('codellama/CodeLlama-7b-Instruct-hf', model_max_length=4096)
    _tokenizer.pad_token = _tokenizer.eos_token
    _tokenizer.padding_side = "right"

    print(_tokenizer.model_max_length)
    codellama_template = CodellamaTemplate()
    algo_dataset = AlgoDataset(data_path="../data/algorithm.json",
                               batch_size=4,
                               tokenizer=_tokenizer,
                               prompt_template=codellama_template)

    dataloader = DataLoader(dataset=algo_dataset, batch_size=4, shuffle=False)

    for batch in dataloader:
        print(batch)
        print(batch['labels'].size())
        break

    from transformers import AutoTokenizer, DataCollatorWithPadding
