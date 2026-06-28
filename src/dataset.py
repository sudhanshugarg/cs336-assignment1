import torch
from torch.utils.data import Dataset
from tokenizer import Tokenizer

class TextFileReader(Dataset):
    def __init__(self, path: str, tokenizer: Tokenizer, seq_length: int = 32):
        self.seq_length = seq_length
        self.tokenizer = tokenizer

        with open(path, "r") as f:
            self.raw_data = f.read()
            self.tokens, self.token_ints = self.tokenizer.tokenize([self.raw_data])[0]
            f.close()
        self.n = len(self.token_ints)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:

        res = torch.tensor(self.token_ints[i:i+self.seq_length]), torch.tensor(self.token_ints[i+1:i+self.seq_length+1])
        # print(f"{i}: {res}")
        return res

    def __len__(self) -> int:
        return self.n - self.seq_length