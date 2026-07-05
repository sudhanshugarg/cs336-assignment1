from typing import Any
from cs336_basics.pretokenization_example import find_chunk_boundaries
import regex
from collections import defaultdict
import heapq

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
        self.merges = []
        self.tokenMap = {}
        self.nextTokenId = 256

    def train_bpe(self) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        """
        1. pre tokenization - get words
        2. for each word, count frequency of pairs
        3. take best pair
        """
        return ({}, [])

    def breakup_1(self):
        with open(self.input_path, "rb") as f:
            for start, end in zip(self.boundaries[:-1], self.boundaries[1:]):
                f.seek(start)
                chunk = f.read(end - start).decode("utf-8", errors="ignore")
                for match in PAT.finditer(chunk):
                    word_int_tuple = list(match.group().encode("utf-8"))
                    word_tuple_arr = []
                    for c in word_int_tuple:
                        self.tokenMap[bytes([c])] = c
                        word_tuple_arr.append(bytes([c]))
                    word_tuple = tuple(word_tuple_arr)
                    self.words[word_tuple] += 1

    def pairwise_2(self):
        #next, for each word, we go pairwise
        #take each pairwise tuple, and put into a heap.
        #find the max one.
        #join those two at all positions
        #add these two the result.
        h = []
        freq = defaultdict(lambda: 0)
        where = defaultdict(lambda: [])
        for word_tuple, count in self.words.items():
            n = len(word_tuple)
            for i in range(n-1):
                pair = (word_tuple[i], word_tuple[i+1])
                freq[pair] += count
                where[pair].append(word_tuple)

        for pair, count in freq.items():
            heapq.heappush(h, (-count, pair))
        
        while (len(h) > 0) and self.nextTokenId < self.vocab_size:
            largest, tokenPair = heapq.heappop(h)
            largest *= -1
            self.merges.append(tokenPair)
            nextToken = tokenPair[0] + tokenPair[1]
            self.tokenMap[nextToken] = self.nextTokenId
            self.nextTokenId += 1

            #need to update the self.words
            #for that, i need to know, for this token, all the words it was in.
            word_tuples_to_be_changed = where[tokenPair]
            for word_tuple in word_tuples_to_be_changed:
                self.update_counts_and_add_to_heap_3(word_tuple, tokenPair, h)
    
    def update_counts_and_add_to_heap_3(self, 
                                        word_tuple: list[bytes], 
                                        tokenPair: list[bytes], 
                                        h: list[tuple[int, list[bytes]]]) -> None:
        """
        we need to find all occurences of tokenPair in word_tuple, and merge them.
        then, once done, we need to update the counts of all the tokens.
        """

        prevTokenPairCounts = defaultdict(lambda: 0)
        newTokenCounts = defaultdict(lambda: 0)
        newPair = tokenPair[0] + tokenPair[1]

        n = len(word_tuple)
        new_word_tuple = []
        for i in range(n-1):
            currPair = word_tuple[i] + word_tuple[i+1]
            prevTokenPairCounts[currPair] += 1
            if currPair != newPair:
                new_word_tuple.append(word_tuple[i])
            else:
                new_word_tuple.append(newPair)
                i += 1
        
        n2 = len(new_word_tuple)
        for i in range(n2-1):
            currPair = new_word_tuple[i] + new_word_tuple[i+1]
            newTokenCounts[currPair] += 1
        





params = {
    "input_path": "src/resources/words.txt",
    "vocab_size": 100,
    "special_tokens": [
        "<|endoftext|>"
    ]
}

tokenizer = BPETokenizer(**params)
tokenizer.breakup()
