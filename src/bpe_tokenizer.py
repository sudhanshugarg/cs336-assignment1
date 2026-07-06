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
        
        self.words = []
        self.wordCount = defaultdict(lambda: 0)
        self.merges = []
        self.tokenMap = {}
        self.nextTokenId = 256
        self.pairsInHeap = set()
        self.h = []

    def train_bpe(self) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        """
        1. pre tokenization - get words
        2. for each word, count frequency of pairs
        3. take best pair
        """
        return ({}, [])

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
                for match in PAT.finditer(chunk):
                    word = match.group()
                    self.words.append(word)

                    word_int_tuple = list(word.encode("utf-8"))
                    word_tuple_arr = []
                    for c in word_int_tuple:
                        self.tokenMap[bytes([c])] = c
                        word_tuple_arr.append(bytes([c]))
                    word_tuple = tuple(word_tuple_arr)
                    
                    self.wordCount[word_tuple] += 1
                    

    def pairwise_2(self):
        #next, for each word, we go pairwise
        #take each pairwise tuple, and put into a heap.
        #find the max one.
        #join those two at all positions
        #add these two the result.
        
        freq = defaultdict(lambda: 0)
        where = defaultdict(lambda: [])
        for word_tuple, count in self.wordCount.items():
            n = len(word_tuple)
            for i in range(n-1):
                pair = (word_tuple[i], word_tuple[i+1])
                freq[pair] += count
                where[pair].append(word_tuple)

        for pair, count in freq.items():
            heapq.heappush(self.h, (-count, pair))
            self.pairsInHeap.add(pair)
        
        tokenCountsChanged = defaultdict(lambda: 0)

        k = 0
        while (len(self.h) > 0) and self.nextTokenId < self.vocab_size:
            k += 1
            while True:
                largest, tokenPair = heapq.heappop(self.h)
                largest *= -1
                if tokenPair not in tokenCountsChanged:
                    break
                
                updated_count = largest + tokenCountsChanged[tokenPair]
                del tokenCountsChanged[tokenPair]
                heapq.heappush(self.h, (-updated_count, tokenPair))

            self.merges.append(tokenPair)           
            tokenPairJoined = tokenPair[0] + tokenPair[1]
            self.tokenMap[tokenPairJoined] = self.nextTokenId
            self.nextTokenId += 1

            #need to update self.wordCount
            #for that, i need to know, for this token, all the words it was in.
            word_tuples_to_be_changed = where[tokenPair]
            createdTokens = set()
            for word_tuple in word_tuples_to_be_changed:
                self.update_counts(list(word_tuple), self.wordCount[word_tuple], tokenPair, tokenCountsChanged, createdTokens)

            print("next...")
            print(largest, tokenPair, tokenPairJoined)
            word_tuples_decoded = []
            for word_tuple in word_tuples_to_be_changed:
                word_tuples_decoded.append(self._get_str_list(list(word_tuple)))
                    
            print(word_tuples_decoded)
            print("created:", createdTokens)
            # for k, v in self.wordCount.items():
            #     print(k, v)
            self._print_word_count()

            for new_token_pair in createdTokens:
                heapq.heappush(self.h, (-tokenCountsChanged[new_token_pair], new_token_pair))
                self.pairsInHeap.add(new_token_pair)
                del tokenCountsChanged[new_token_pair]
            

    def update_counts(self, 
                        word_tuple: list[bytes],
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

        n = len(word_tuple)
        new_word_tuple = []
        i = 0
        while i < (n-1):
            currPair = (word_tuple[i], word_tuple[i+1])
            prevTokenPairCounts[currPair] += word_tuple_count
            i += 1
        
        i = 0
        while i < (n-1):
            currPair = (word_tuple[i], word_tuple[i+1])
            if currPair != tokenPair:
                new_word_tuple.append(word_tuple[i])
            else:
                new_word_tuple.append(tokenPairJoined)
                i += 1
            i += 1

        if i < n:
            new_word_tuple.append(word_tuple[n-1])


        n2 = len(new_word_tuple)
        for i in range(n2-1):
            currPair = (new_word_tuple[i], new_word_tuple[i+1])
            if currPair not in self.pairsInHeap:
                createdTokens.add((new_word_tuple[i], new_word_tuple[i+1]))
            newTokenCounts[currPair] += word_tuple_count
        
        
        print("prev:", prevTokenPairCounts)
        print("new:", newTokenCounts)


        for pair, count in prevTokenPairCounts.items():
            tokenCountsChanged[pair] -= count
        for pair, count in newTokenCounts.items():
            tokenCountsChanged[pair] += count
            if tokenCountsChanged[pair] == 0:
                del tokenCountsChanged[pair]

        #we've already chosen tokenPair, dont need it anymore.
        del tokenCountsChanged[tokenPair]

        print("changes:", tokenCountsChanged)
        del self.wordCount[tuple(word_tuple)]
        self.wordCount[tuple(new_word_tuple)] = word_tuple_count


params = {
    "input_path": "src/resources/example.txt",
    "vocab_size": 258,
    "special_tokens": [
        "<|endoftext|>"
    ]
}

tokenizer = BPETokenizer(**params)
tokenizer.breakup_1()
# tokenizer._print_word_count()

tokenizer.pairwise_2()