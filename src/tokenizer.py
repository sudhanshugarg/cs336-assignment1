from typing import Iterable, Iterator
import pickle
import heapq

class Tokenizer():
    def __init__(self, 
                 vocab: dict[int, bytes], 
                 merges: list[tuple[bytes, bytes]], 
                 special_tokens: list[str] | None = None):
        self.tokenIntMap = vocab
        self.merges = merges
        if special_tokens:
            self.special_tokens = set(special_tokens)

        self.tokenMap: dict[bytes, int] = {}
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
            data = pickle.load(f)
            tokenIntMap = data["tokenmap"]

        with open(merges_filepath, "rb") as f:
            data = pickle.load(f)
            merges = data["merges"]

        return cls(tokenIntMap, merges, special_tokens)

    def _encode_brute_force(self, text) -> list[int]:
        byte_int_array = list(text.encode("utf-8"))
        bytearray = [bytes([x]) for x in byte_int_array]

        """
        brute force        
        """
        merged = True
        while merged:
            merged = False
            minPriority = len(self.merges)
            minIndex = -1

            n = len(bytearray)
            for i in range(n-1):
                currPair = (bytearray[i], bytearray[i+1])
                if currPair not in self.priority:
                    continue
                merged = True
                currPriority = self.priority[currPair]
                if minPriority > currPriority:
                    minPriority = currPriority
                    minIndex = i

            if merged:
                nextarray = bytearray[:minIndex]
                nextarray.append((bytearray[minIndex] + bytearray[minIndex+1]))
                nextarray.extend(bytearray[minIndex+2:])
                bytearray = nextarray
        
        result = []
        for i in range(len(bytearray)):
            result.append(self.tokenMap[bytearray[i]])

        return result

    def encode(self, text) -> list[int]:
        return self._encode_brute_force(text)

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        pass

    def decode(self, ids: list[int]) -> str:
        ids_bytes = [self.tokenIntMap[x] for x in ids]
        return b''.join(ids_bytes).decode("utf-8")


tokenizer = Tokenizer.from_files(
                vocab_filepath="src/resources/vocab_next_50.pkl",
                merges_filepath="src/resources/vocab_next_50.pkl"
            )


examples = [
    "hello",
    "world",
    "whats going on",
    "these days"
]

for e in examples:
    encoded = tokenizer.encode(e)
    decoded = tokenizer.decode(encoded)
    print(e, encoded, decoded)
    assert e == decoded
