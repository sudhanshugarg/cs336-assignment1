from torch.utils.data import Dataset

class TextFileReader(Dataset):
    def __init__(self, path: str, seq_length: int = 32):
        with open(path, "r") as f:
            self.data = "whats up old man, how goes it these days, what are you up to"
            f.close()
        self.seq_length = seq_length
        self.n = len(self.data)

    def __getitem__(self, i: int) -> tuple[str, str]:
        return self.data[i:i+self.seq_length], self.data[i+1:i+self.seq_length+1]

    def __len__(self) -> int:
        return self.n - self.seq_length