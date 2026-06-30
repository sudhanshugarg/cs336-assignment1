from abc import ABC
from collections import defaultdict
import heapq
import pickle
import os

class Tokenizer(ABC):
    padding_token = "<PAD>"
    padding_token_int = 0

    def __init__(self, 
                 corpus_file_path: str, 
                 tokenizer_path: str, 
                 vocab_size: int = 1000, 
                 overwrite: bool = False) -> None:
        super().__init__()
        self.tokenMap = {}
        self.tokenMapInt = {}
        self.tokenPriority = {}
        self.minFreq = 1

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
        self.tokenPositions = defaultdict(lambda: set())
        self.freqCounter = defaultdict(lambda: 0)
        self.vocabSize = vocab_size

        self._create_tokens(self.vocabSize)
        self._store_tokens(tokenizer_path)


    def _read_corpus(self):
        with open(self.file_path, "r") as f:
            self.corpus = f.read(-1)
        print(f"corpus length = {len(self.corpus)}")
        

    def _create_tokens(self, vocab_size: int):
        # create byte pair encoding for tokens
        
        # initializing tokens
        chars = list(self.corpus)
        unique_chars = sorted(set(chars))
        i = self.padding_token_int + 1
        for c in unique_chars:
            self.tokenMap[c] = i
            self.tokenMapInt[i] = c
            self.tokenPriority[c] = vocab_size #lowest priority for single chars
            i += 1


        unique_chars.append(self.padding_token)
        self.tokenMap[self.padding_token] = self.padding_token_int
        self.tokenMapInt[self.padding_token_int] = self.padding_token
        self.tokenPriority[self.padding_token] = vocab_size

        self.unique_chars = unique_chars

        for i in range(len(self.corpus)):
            token = self.corpus[i]
            self.tokenPositions[token].add(i)

        # count pairs of tokens, and put them into a heap
        for i in range(len(self.corpus) - 1):
            t1 = self.corpus[i]
            t2 = self.corpus[i+1]
            token = f"{t1}{t2}"
            self.freqCounter[token] += 1
            self.tokenPositions[token].add(i)

        h = []
        exists_in_heap = set()
        for token, value in self.freqCounter.items():
            if value > self.minFreq:
                heapq.heappush(h, (-value, token)) # max heap
                exists_in_heap.add(token)
    
        #start adding tokens
        priority = self.padding_token_int + 1
        num_elements = len(self.unique_chars)
        while len(h) > 0 and num_elements < vocab_size:
            #do your thang!
            freq, token = heapq.heappop(h)
            if priority % 500 == 0:
                print(f"token at {priority} = {token[:10]}, len(tokenMap) = {len(self.tokenMap)}")
            # print(f"choosing token #{token}# with frequency #{-freq}#")
            # now i have the next token
            p = len(self.tokenMap)
            self.tokenMap[token] = p
            self.tokenMapInt[p] = token
            self.tokenPriority[token] = priority
            priority += 1
            num_elements += 1
            #now, i need to find all the positions for this token
            positions = sorted(self.tokenPositions[token])
            # print(f"for token #{token}#, no. of positions = {len(positions)}, first 5 = {positions[0:5]}")

            # for each position, i need to pair it with ALL 
            # the next tokens it has
            #
            newTokensGenerated = set()
            for i in range(len(positions)):
                # if i > 5:
                #     break

                pos = positions[i]
                pairedTokenStart = pos + len(token)
                # print(f"\nstarting at position {pos} for token #{token}#")
                for j in range(pairedTokenStart+1, self.N):
                    pairedToken = self.corpus[pairedTokenStart:j]
                    if pairedToken not in self.tokenMap:
                        # print(f"didn't find #{pairedToken}# in tokenMap, going to next pos")
                        break
                    nextToken = f"{token}{pairedToken}"
                    # print(f"paired token for #{token}# is #{pairedToken}#, together: {nextToken}")
                    if nextToken in exists_in_heap:
                        continue

                    if nextToken not in newTokensGenerated:
                        newTokensGenerated.add(nextToken)
                    # self.freqCounter[nextToken] += 1
                    # print(f"found new token #{nextToken}# at position #{pos}#")
                    self.tokenPositions[nextToken].add(pos)

            # newTokensDict = {oneMoreToken: self.tokenPositions[oneMoreToken] for oneMoreToken in newTokensGenerated}
            # print(f"all new tokens:\n: {newTokensDict}")
            for newToken in newTokensGenerated:
                self.freqCounter[newToken] = len(self.tokenPositions[newToken])
                if self.freqCounter[newToken] > self.minFreq:
                    heapq.heappush(h, (-self.freqCounter[newToken], newToken))
                    exists_in_heap.add(newToken)


    def _store_tokens(self, tokenizer_path: str):
        with open(tokenizer_path, "wb") as f:
            tokenData = {
                "tokenMap": self.tokenMap,
                "tokenMapInt": self.tokenMapInt,
                "tokenPriority": self.tokenPriority
            }
            pickle.dump(tokenData, f)


    def tokenize(self, inputs: list[str], seq_length: int) -> list[tuple[list[str], list[int]]]:
        return [self._tokenize_and_pad(input, seq_length) for input in inputs]


    def _tokenize_and_pad(self, input: str, seq_length: int) -> tuple[list[str], list[int]]:
        tokens, tokenInts = self._tokenize_single_input(input)
        n = len(tokenInts)
        if n < seq_length:
            tokens.extend([self.padding_token] * (seq_length - n))
            tokenInts.extend([self.padding_token_int] * (seq_length - n))

        return tokens, tokenInts


    def _tokenize_single_input(self, input: str) -> tuple[list[str], list[int]]:
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
                heapq.heappush(tokenCandidateInts, (self.tokenPriority[tokenCandidate], self.tokenMap[tokenCandidate], tokenCandidate))
                end = j
            
            tokenCandidates = [tokenInt[2] for tokenInt in tokenCandidateInts]
            if len(tokenCandidateInts) > 0:
                # take the token with highest freq (i.e. lowest int value)
                _, tokenInt, token = heapq.heappop(tokenCandidateInts)
                tokenInts.append(tokenInt)
                tokens.append(token)
                # print(f"got candidate tokens #{tokenCandidates}#, choosing #{token}#")
                i += len(token)
            else:
                # print(f"got no tokens starting from {i}")
                i += 1

        return tokens, tokenInts

vocabSize = 10000
tokenizer_path = f"src/resources/tokenMap_{vocabSize}.pkl"
tokenizer = Tokenizer("src/resources/input.txt", tokenizer_path, vocab_size=vocabSize, overwrite=False)
with open(tokenizer_path, "rb") as f:
    tmap = pickle.load(f)

lengths = [(len(key), key) for key in tmap["tokenMap"].keys()]
lengths.sort()
print(lengths[-3:])
# for k, v in tmap.items():
#     print(f"#{k}#: -{v}-")

s = "thereit is the best of the best"
t, ti = tokenizer._tokenize_and_pad(s, 16)
print(t)
print(ti)
