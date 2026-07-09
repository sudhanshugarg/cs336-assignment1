from typing import Iterable, Iterator
import pickle

class Tokenizer():
    def __init__(self, 
                 vocab: dict[int, bytes], 
                 merges: list[tuple[bytes, bytes]], 
                 special_tokens: list[str] | None = None):
        self.tokenIntMap = vocab
        self.merges = merges
        if special_tokens:
            self.special_tokens = set(special_tokens)

        self.tokenMap = {}
        self._process_vocab()
        self.priority = {}
        self._process_merges()

    def _process_vocab(self):
        for key, value in self.tokenIntMap.items():
            self.tokenMap[value] = key

    def _process_merges(self):
        #create mapping from merges to index and back
        for i, merge in enumerate(self.merges):
            self.priority[merge] = i


    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        with open(vocab_filepath, "rb") as f:
            data = pickle.load(vocab_filepath)
            tokenIntMap = data["tokenmap"]

        with open(merges_filepath, "rb") as f:
            data = pickle.load(merges_filepath)
            merges = data["merges"]

        return cls(tokenIntMap, merges, special_tokens)

    def encode(self, text) -> list[int]:
        pass

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        pass

    def decode(self, ids: list[int]) -> str:
        pass

