from __future__ import annotations
from typing import Any
from cs336_basics.pretokenization_example import find_chunk_boundaries
import regex
from collections import defaultdict
from multiprocessing import Process
import heapq


class TokenPair():
    def __init__(self, count: int, pair: tuple[bytes, bytes]):
        self.count = count
        self.pair = pair
    
    def __lt__(self, other: TokenPair):
        if self.count > other.count:
            return True
        elif self.count == other.count:
            return self.pair > other.pair
        else:
            return False
        
    def __hash__(self):
        return (hash(self.count) ^ hash(self.pair)) % 1000000007
    
    def __eq__(self, other: TokenPair):
        return self.count == other.count and self.pair == other.pair


class BPETokenizer():
    def __init__(self, *args: Any, **kwargs: Any):
        self.input_path = kwargs["input_path"]
        self.vocab_size = kwargs["vocab_size"]
        self.special_tokens = kwargs["special_tokens"]
        special_token_bytes = self.special_tokens[0].encode("utf-8")
        special_token_regex = regex.escape(self.special_tokens[0])
        regex_string = r"""'(?:[sdmt]|ll|ve|re)| ?""" + special_token_regex + r"""| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        # print(regex_string)
        self.PAT = regex.compile(regex_string)


        with open(self.input_path, "rb") as f:
            num_processes = 3
            self.boundaries = find_chunk_boundaries(f, num_processes, special_token_bytes)
        
        # print(self.boundaries)

        self.wordCount = defaultdict(lambda: 0)
        self.merges = []
        self.tokenMap = {}
        self.tokenMapInt = {}
        self.nextTokenId = 257
        self.pairsInHeap = set()
        self.h = []
        self.whereIsPair = defaultdict(lambda: set())
        self.tokenFreq = defaultdict(lambda: 0)


        self.tokenMap[special_token_bytes] = 0
        self.tokenMapInt[0] = special_token_bytes
        for i in range(256):
            self.tokenMap[bytes([i])] = i+1
            self.tokenMapInt[i+1] = bytes([i])

    def train_bpe(self) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        """
        1. pre tokenization - get words
        2. for each word, count frequency of pairs
        3. take best pair
        """
        self.breakup_1()
        self.pairwise_2()
        return (self.tokenMapInt, self.merges)

    def _get_str_list(self, word: list[bytes]) -> list[str]:
        result = []
        for w in word:
            result.append(w.decode("utf-8"))
        return result

    def _print_word_count(self):
        d = {}
        for k, v in self.wordCount.items():
            d[tuple(self._get_str_list(k))] = v
        print(d)

    def breakup_1(self):
        with open(self.input_path, "rb") as f:
            for start, end in zip(self.boundaries[:-1], self.boundaries[1:]):
                f.seek(start)
                chunk = f.read(end - start).decode("utf-8", errors="ignore")
                for match in self.PAT.finditer(chunk):
                    word = match.group()
                    if word == self.special_tokens[0]:
                        continue

                    word_int_tuple = list(word.encode("utf-8"))
                    word_tuple_arr = []
                    for c in word_int_tuple:
                        # self.tokenMap[bytes([c])] = c
                        # self.tokenMapInt[c] = bytes([c])
                        word_tuple_arr.append(bytes([c]))
                    word_tuple = tuple(word_tuple_arr)
                    
                    self.wordCount[word_tuple] += 1
                    

    def pairwise_2(self):
        #next, for each word, we go pairwise
        #take each pairwise tuple, and put into a heap.
        #find the max one.
        #join those two at all positions
        #add these two the result.

        for word_tuple, count in self.wordCount.items():
            n = len(word_tuple)
            for i in range(n-1):
                pair = (word_tuple[i], word_tuple[i+1])
                self.tokenFreq[pair] += count
                self.whereIsPair[pair].add(word_tuple)

        for pair, count in self.tokenFreq.items():
            heapq.heappush(self.h, TokenPair(count, pair))
            self.pairsInHeap.add(pair)

        tokenCountsChanged = defaultdict(lambda: 0)

        k = 0
        while (len(self.h) > 0) and self.nextTokenId < self.vocab_size:
            k += 1
            while True:
                item = heapq.heappop(self.h)
                largest, tokenPair = item.count, item.pair
                if tokenPair not in tokenCountsChanged:
                    break
                
                updated_count = largest + tokenCountsChanged[tokenPair]
                del tokenCountsChanged[tokenPair]
                heapq.heappush(self.h, TokenPair(updated_count, tokenPair))

            # print("\n\nnext...")
            self.tokenFreq[tokenPair] = largest
            tokenPairJoined = tokenPair[0] + tokenPair[1]
            # print(largest, tokenPair, tokenPairJoined)

            self.merges.append(tokenPair)           
            self.tokenMap[tokenPairJoined] = self.nextTokenId
            self.tokenMapInt[self.nextTokenId] = tokenPairJoined
            self.nextTokenId += 1

            #need to update self.wordCount
            #for that, i need to know, for this token, all the words it was in.
            word_tuples_to_be_changed = self.whereIsPair[tokenPair].copy()
            # print("to be changed:", word_tuples_to_be_changed)
            createdTokens = set()
            for word_tuple in word_tuples_to_be_changed:
                # print("calling with: ", word_tuple)
                self.update_counts(list(word_tuple), self.wordCount[word_tuple], tokenPair, tokenCountsChanged, createdTokens)

            # word_tuples_decoded = []
            # for word_tuple in word_tuples_to_be_changed:
            #     word_tuples_decoded.append(self._get_str_list(list(word_tuple)))
                    
            # print(word_tuples_decoded)
            # print("created:", createdTokens)
            # self._print_word_count()

            for new_token_pair in createdTokens:
                heapq.heappush(self.h, TokenPair(tokenCountsChanged[new_token_pair], new_token_pair))
                self.pairsInHeap.add(new_token_pair)
                del tokenCountsChanged[new_token_pair]
            

    def update_counts(self, 
                        word_tuple_list: list[bytes],
                        word_tuple_count: int,
                        tokenPair: tuple[bytes, bytes],
                        tokenCountsChanged: dict[tuple[bytes, bytes], int],
                        createdTokens: set[tuple[bytes, bytes]]) -> None:
        """
        we need to find all occurences of tokenPair in word_tuple, and merge them.
        then, once done, we need to update the counts of all the tokens.
        """

        prevTokenPairCounts = defaultdict(lambda: 0)
        newTokenCounts = defaultdict(lambda: 0)
        tokenPairJoined = tokenPair[0] + tokenPair[1]

        n = len(word_tuple_list)
        word_tuple = tuple(word_tuple_list)
        new_word_tuple_list = []
        i = 0
        while i < (n-1):
            currPair = (word_tuple_list[i], word_tuple_list[i+1])
            prevTokenPairCounts[currPair] += word_tuple_count
            if word_tuple in self.whereIsPair[currPair]:
                self.whereIsPair[currPair].remove(word_tuple)
            i += 1
        
        i = 0
        while i < (n-1):
            currPair = (word_tuple_list[i], word_tuple_list[i+1])
            if currPair != tokenPair:
                new_word_tuple_list.append(word_tuple_list[i])
            else:
                new_word_tuple_list.append(tokenPairJoined)
                i += 1
            i += 1

        if i < n:
            new_word_tuple_list.append(word_tuple_list[n-1])


        new_word_tuple = tuple(new_word_tuple_list)
        self.wordCount[new_word_tuple] = word_tuple_count
        del self.wordCount[word_tuple]

        n2 = len(new_word_tuple_list)
        for i in range(n2-1):
            currPair = (new_word_tuple_list[i], new_word_tuple_list[i+1])
            if currPair not in self.pairsInHeap:
                createdTokens.add(currPair)
            self.whereIsPair[currPair].add(new_word_tuple)
            newTokenCounts[currPair] += word_tuple_count
        
        # print("prev:", prevTokenPairCounts)
        # print("new:", newTokenCounts)


        for pair, count in prevTokenPairCounts.items():
            tokenCountsChanged[pair] -= count
        for pair, count in newTokenCounts.items():
            tokenCountsChanged[pair] += count
            if tokenCountsChanged[pair] == 0:
                del tokenCountsChanged[pair]

        #we've already chosen tokenPair, dont need it anymore.
        del tokenCountsChanged[tokenPair]
        # print("changes:", tokenCountsChanged)


params = {
    "input_path": "tests/fixtures/tinystories_sample_5M.txt",
    "vocab_size": 1000,
    "special_tokens": [
        "<|endoftext|>"
    ]
}

tokenizer = BPETokenizer(**params)
tokenizer.train_bpe()
print(tokenizer.tokenFreq[(b"\n", b"\n")])
print(tokenizer.tokenFreq[(b" g", b"ive")])
# tokenizer._print_word_count()

# import pickle
# with open("tests/_snapshots/test_train_bpe_special_tokens.pkl", "rb") as f:
#     data = pickle.load(f)
#     print(len(data["merges"]))

# import re
# with open("tests/fixtures/tinystories_sample_5M.txt", "r") as f:
#     content = f.read()
#     matches1 = re.findall(r" give", content)
#     print(" give", len(matches1))
#     matches2 = re.findall(r"\n\n", content)
#     print("newline", len(matches2))

from datasets import load_dataset

# ds = load_dataset("roneneldan/TinyStories")
ds2 = load_dataset("Skylion007/openwebtext")

