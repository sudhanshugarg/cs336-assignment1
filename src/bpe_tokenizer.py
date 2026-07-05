from typing import Any

class BPETokenizer():
    def __init__(self, *args: Any, **kwargs: Any):
        self.input_path = kwargs["input_path"]
        self.vocab_size = kwargs["vocab_size"]
        self.special_tokens = kwargs["special_tokens"]

    def train_bpe(self) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        """
        1. pre tokenization - get words
        2. for each word, count frequency of pairs
        3. take best pair
        """
