from datasets import load_dataset, Dataset, load_from_disk

ds = load_dataset("roneneldan/TinyStories", split="train")
# ds = Dataset.from_file("/Users/sudgarg/.cache/huggingface/datasets/roneneldan___tiny_stories/default/0.0.0/f54c09fd23315a6f9c86f9dc80f725de7d8f9c64/tiny_stories-train-00001-of-00004.arrow")

train_300k = ds.select(range(300_000))

with open("src/resources/tinystories_300k.bin", "w") as f:
    for row in train_300k:
        text = row["text"].strip()
        f.write(f"{text}\n")

print(ds.column_names)