from abc import ABC
from collections import defaultdict
import heapq
import pickle
import os

class Tokenizer(ABC):
    def __init__(self, 
                 corpus_file_path: str, 
                 tokenizer_path: str, 
                 vocab_size: int = 1000, 
                 overwrite: bool = False) -> None:
        super().__init__()
        self.tokenMap = {}
        self.tokenMapInt = {}
        self.tokenPriority = {}

        if not overwrite and os.path.exists(tokenizer_path):
            with open(tokenizer_path, "rb") as f:
                tokenData = pickle.load(f)
                self.tokenMap = tokenData["tokenMap"]
                self.tokenMapInt = tokenData["tokenMapInt"]
                self.tokenPriority = tokenData["tokenPriority"]
            return

        if not os.path.exists(corpus_file_path):
            raise ValueError(f"input file {corpus_file_path} does not exist")

        self.file_path = corpus_file_path
        self.corpus = ""
        self._read_corpus()
        self.N = len(self.corpus)
        self.tokenPositions = defaultdict(lambda: [])
        self.freqCounter = defaultdict(lambda: 0)
        self.vocabSize = vocab_size

        self.create_tokens(self.vocabSize)
        self.store_tokens(tokenizer_path)


    def _read_corpus(self):
        with open(self.file_path, "r") as f:
            self.corpus = f.read(-1)
        print(f"corpus length = {len(self.corpus)}")
        

    def create_tokens(self, vocab_size: int):
        # create byte pair encoding for tokens
        
        # initializing tokens
        chars = list(self.corpus)
        unique_chars = set(chars)
        self.unique_chars = unique_chars
        i = 0
        for c in unique_chars:
            self.tokenMap[c] = i
            self.tokenMapInt[i] = c
            self.tokenPriority[c] = vocab_size #lowest priority for single chars
            i += 1

        for i in range(len(self.corpus)):
            token = self.corpus[i]
            self.tokenPositions[token].append(i)

        # count pairs of tokens, and put them into a heap
        for i in range(len(self.corpus) - 1):
            t1 = self.corpus[i]
            t2 = self.corpus[i+1]
            token = f"{t1}{t2}"
            self.freqCounter[token] += 1
            self.tokenPositions[token].append(i)

        h = []
        for key, value in self.freqCounter.items():
            heapq.heappush(h, (-value, key)) # max heap
    
        #start adding tokens
        priority = 1
        while len(h) > 0 and len(self.tokenMap) < vocab_size:
            #do your thang!
            freq, token = heapq.heappop(h)
            # print(f"choosing token #{token}# with frequency #{-freq}#")
            # now i have the next token
            p = len(self.tokenMap)
            self.tokenMap[token] = p
            self.tokenMapInt[p] = token
            self.tokenPriority[token] = priority
            priority += 1
            #now, i need to find all the positions for this token
            positions = self.tokenPositions[token]
            # print(f"for token #{token}#, no. of positions = {len(positions)}, first 5 = {positions[0:5]}")

            # for each position, i need to pair it with ALL 
            # the next tokens it has
            #
            newTokensGenerated = set()
            for i in range(len(positions)):
                if i > 2:
                    break

                pos = positions[i]
                pairedTokenStart = pos + len(token)
                for j in range(pairedTokenStart+1, self.N):
                    pairedToken = self.corpus[pairedTokenStart:j]
                    # print(f"paired token for #{token}# is #{pairedToken}#")
                    if pairedToken not in self.tokenMap:
                        break

                    nextToken = f"{token}{pairedToken}"
                    # print(f"considering next token as {nextToken}")
                    if nextToken not in newTokensGenerated:
                        newTokensGenerated.add(nextToken)
                    self.freqCounter[nextToken] += 1
                    self.tokenPositions[nextToken].append(i)

            for newToken in newTokensGenerated:
                heapq.heappush(h, (-self.freqCounter[newToken], newToken))


    def store_tokens(self, tokenizer_path: str):
        with open(tokenizer_path, "wb") as f:
            tokenData = {
                "tokenMap": self.tokenMap,
                "tokenMapInt": self.tokenMapInt,
                "tokenPriority": self.tokenPriority
            }
            pickle.dump(tokenData, f)


    def tokenize(self, inputs: list[str]) -> list[tuple[list[str], list[int]]]:
        return [self.tokenize_single_input(input) for input in inputs]


    def tokenize_single_input(self, input: str) -> tuple[list[str], list[int]]:
        i = 0
        n = len(input)
        # print(f"trying to tokenize: #{input}, current tokens = {len(self.tokenMap)}#")
        tokens = []
        tokenInts = []
        while i < n:
            end = i
            j = i
            tokenCandidateInts = []
            while j < n:
                j += 1
                tokenCandidate = input[i:j]
                if input[i:j] not in self.tokenMap:
                    break
                heapq.heappush(tokenCandidateInts, (self.tokenPriority[tokenCandidate], self.tokenMap[tokenCandidate]))
                end = j
            
            # print(f"got token {token}")
            if len(tokenCandidateInts) > 0:
                # take the token with highest freq (i.e. lowest int value)
                _, tokenInt = heapq.heappop(tokenCandidateInts)
                tokenInts.append(tokenInt)
                tokens.append(self.tokenMapInt[tokenInt])
                i = end
            else:
                print(f"got no tokens starting from {i}")
                i += 1

        return tokens, tokenInts

# vocabSize = 100
# tokenizer_path = f"src/resources/tokenMap_{vocabSize}.pkl"
# tokenizer = Tokenizer("src/resources/input.txt", tokenizer_path, vocab_size=vocabSize)
# with open(tokenizer_path, "rb") as f:
#     tmap = pickle.load(f)

# for k, v in tmap.items():
#     print(f"#{k}#: -{v}-")

# s = "thereit is the best of the best"
# t, ti = tokenizer.tokenize_single_input(s)
# print(t)
# print(ti)
