from typing import Iterable, Iterator
import pickle
import regex
from functools import cmp_to_key

class Tokenizer():
    PreTokenizer = regex.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

    def token_sort(self, word1, word2):
        if word1 == word2:
            return 0

        if word1.startswith(word2):
            return -1
        elif word2.startswith(word1):
            return 1

        if word1 < word2:
            return -1
        return 1

    def __init__(self, 
                 vocab: dict[int, bytes], 
                 merges: list[tuple[bytes, bytes]], 
                 special_tokens: list[str] | None = None):
        self.tokenIntMap = vocab
        self.merges = merges

        self.tokenMap: dict[bytes, int] = {}
        self._process_vocab()
        self.priority = {}
        self._process_merges()

        self.special_tokens = set()
        if special_tokens:
            special_tokens.sort(key=cmp_to_key(self.token_sort))
            print(special_tokens)
            #() braces to ensure that the special tokens are also covered
            self.special_tokens_regex = f"({"|".join([regex.escape(x) for x in special_tokens])})"

            #find next vocab index that can be used
            next_index = max(self.tokenIntMap.keys()) + 1
            self.special_tokens = set(special_tokens)
            for special_token in self.special_tokens:
                special_token_bytes = special_token.encode("utf-8")
                if special_token_bytes not in self.tokenMap:
                    self.tokenMap[special_token_bytes] = next_index
                    self.tokenIntMap[next_index] = special_token_bytes
                    next_index += 1

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

    def _encode_long_string(self, text) -> list[int]:
        if self.special_tokens:
            texts = regex.split(self.special_tokens_regex, text)
        else:
            texts = [text]

        encoded_array = []
        for text in texts:
            if text in self.special_tokens:
                encoded_array.append(self.tokenMap[text.encode("utf-8")])
            else:
                for match in self.PreTokenizer.finditer(text):
                    small_phrase = match.group()
                    small_phrase_encoded = self._encode_brute_force(small_phrase)
                    encoded_array.extend(small_phrase_encoded)
        return encoded_array

    def _encode_long_string_generator(self, text) -> Iterator[int]:
        if self.special_tokens:
            texts = regex.split(self.special_tokens_regex, text)
        else:
            texts = [text]

        for text in texts:
            if text in self.special_tokens:
                yield self.tokenMap[text.encode("utf-8")]
            else:
                for match in self.PreTokenizer.finditer(text):
                    small_phrase = match.group()
                    small_phrase_encoded = self._encode_brute_force(small_phrase)
                    for small_phrase_int in small_phrase_encoded:
                        yield small_phrase_int

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
        return self._encode_long_string(text)

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        #the idea, is to keep reading tokens. if you get a token that cannot be merged,
        #then you yield the previous one
        for text in iterable:
            res:Iterator[int] = self._encode_long_string_generator(text)
            yield from res


    def decode(self, ids: list[int]) -> str:
        ids_bytes = [self.tokenIntMap[x] for x in ids]
        return b''.join(ids_bytes).decode("utf-8", errors="replace")


# tokenizer = Tokenizer.from_files(
#                 vocab_filepath="src/resources/vocab_next_50.pkl",
#                 merges_filepath="src/resources/vocab_next_50.pkl",
#                 special_tokens=["<|hey|>"]
#             )

# examples = [
#     "thesemondaytuesdayand<|hey|>what<|hey|><|hey|>no oneis there",
#     "thesemondaytuesdayand",
#     "<|hey|>"
#     # " troubles allover thesemondaytuesdayand friadys",
#     # "these days supercalifragilisticexpialidocisous organiziatioal"
# ]

# for e in examples:
#     print("\nstarting with: ", e)
#     encoded = tokenizer.encode(e)
#     decoded = tokenizer.decode(encoded)
#     print("result: ", encoded, decoded)
#     # assert e == decoded
