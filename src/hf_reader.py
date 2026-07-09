from datasets import load_dataset, Dataset, load_from_disk

ds = load_dataset("roneneldan/TinyStories", split="train")
train_300k = ds.select(range(300_000))

print(len(ds))
endToken = "<|endoftext|>"
with open("src/resources/tinystories_full.bin", "w") as f:
    for row in ds:
        text = row["text"].strip()
        f.write(f"{text}\n{endToken}\n")

print(ds.column_names)