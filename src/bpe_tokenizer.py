from typing import Any
from cs336_basics.pretokenization_example import find_chunk_boundaries
import regex
from collections import defaultdict

PAT = regex.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

class BPETokenizer():
    def __init__(self, *args: Any, **kwargs: Any):
        self.input_path = kwargs["input_path"]
        self.vocab_size = kwargs["vocab_size"]
        self.special_tokens = kwargs["special_tokens"]

        with open(self.input_path, "rb") as f:
            num_processes = 1
            self.boundaries = find_chunk_boundaries(f, num_processes, self.special_tokens[0].encode("utf-8"))
        
        self.words = defaultdict(lambda: 0)

    def train_bpe(self) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        """
        1. pre tokenization - get words
        2. for each word, count frequency of pairs
        3. take best pair
        """
        return ({}, [])

    def breakup(self):
        with open(self.input_path, "rb") as f:
            for start, end in zip(self.boundaries[:-1], self.boundaries[1:]):
                f.seek(start)
                chunk = f.read(end - start).decode("utf-8", errors="ignore")
                for match in PAT.finditer(chunk):
                    word_int_tuple = tuple(match.group().encode("utf-8"))
                    word_tuple = tuple([bytes([c]) for c in word_int_tuple])
                    self.words[word_tuple] += 1
            
        
        #next, for each word, we go pairwise
        for key in self.words.keys():
            print(f"{key}: {self.words[key]}")


    

params = {
    "input_path": "src/resources/words.txt",
    "vocab_size": 100,
    "special_tokens": [
        "<|endoftext|>"
    ]
}

tokenizer = BPETokenizer(**params)
tokenizer.breakup()
