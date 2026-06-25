# BPE Tokenizer Learning Guide

## What is Byte Pair Encoding (BPE)?

**Definition:**
Its a compression technique. Given a corpus of words, compress it into a set of tokens that can represent all the words and sentences in that language.


**Why is it needed?**
Languages have a lot of words. Encoding all the words as tokens, would make the vocabulary size too large. To solve this, we create a smaller vocab size.

## The BPE Algorithm

### High-Level Steps:
1. split corpus into bytes (characters). Count frequencies: this is the smallest vocab size you can have. This is your initial token set.
2. Now, count frequencies of frequently occuring pairs of tokens. Take the top pair by frequency, and make that a new token.
3. keep repeating until you reach a desired number of tokens.

### Detailed Breakdown:

**Initialization:**
[Describe the starting state]

**The Merging Process:**
[Explain how pairs are selected and merged]

**Termination Condition:**
[When do you stop merging?]

## Building a Vocabulary

**What gets stored in the vocabulary?**
[Describe what information is kept]

**How many tokens are in a typical BPE vocabulary?**
[Research and note typical sizes]

**How does vocabulary size relate to the number of merge operations?**
[Think about the relationship]

## Encoding and Decoding

**Encoding (text → token IDs):**
[Explain the process of converting raw text to tokens]

**Decoding (token IDs → text):**
[Explain reversing the process]

**What happens with words the tokenizer hasn't seen before?**
[How are unknown or rare words handled?]

## Properties and Trade-offs

**Advantages:**
- [List key benefits]

**Limitations or considerations:**
- [List potential drawbacks]

## Key Insights

[Add any other important observations about BPE as you learn]
