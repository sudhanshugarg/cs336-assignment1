from cs336_basics.pretokenization_example import find_chunk_boundaries
from src.tokenizer import Tokenizer
import numpy as np

vocab_path = "src/resources/openwebtext_vocab_32000_1000.pkl"
input_text_path = "src/resources/openwebtext_sample_0.bin"
output_tokens_path_1 = "src/resources/openwebtext_full_tokenized.bin"
output_tokens_path_np = "src/resources/openwebtext_full_tokenized"

tokenizer = Tokenizer.from_files(
                vocab_filepath=vocab_path,
                merges_filepath=vocab_path,
                special_tokens=["<|endoftext|>"]
            )


with open(input_text_path, "rb") as f:
    num_processes = 1000
    boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
    # print(boundaries)

    output_arr = np.memmap(
        output_tokens_path_1,
        dtype=np.uint16,
        mode="w+",
        shape=(boundaries[-1],)
    )
    # The following is a serial implementation, but you can parallelize this
    # by sending each start/end pair to a set of processes.
    i = 0
    for j, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        if ((j+1) < 391):
            continue
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
        encoded_tokens = np.array(tokenizer.encode(chunk), dtype=np.uint16)
        output_arr[i:i+len(encoded_tokens)] = encoded_tokens
        i += len(encoded_tokens)
        print(f"got {len(encoded_tokens)} from {j+1}th chunk")
        output_arr.flush()

    length = i
    print(f"num_tokens = {length}")

arr2 = np.memmap(
    output_tokens_path_1,
    dtype=np.uint16,
    mode="r",
    shape=(length,)
)

np.save(output_tokens_path_np, arr2)
print(len(arr2))