from datasets import load_dataset, Dataset, load_from_disk

def get_sample_records(ds, n: int = 1000):
    # records = ds.select(range(n))
    print(ds.column_names)
    # print(len(ds))
    records = ds.take(n)
    return records

def write_sample_records(ds, path):
    endToken = "<|endoftext|>"
    with open(path, "w") as f:
        for row in ds:
            text = row["text"].strip()
            f.write(f"{text}\n{endToken}\n")

def generate_tinystores():
    ds = load_dataset("roneneldan/TinyStories", split="train")
    n = 1000
    ds = get_sample_records(ds, n)
    write_sample_records(ds, f"src/resources/tinystories_sample_{n}.bin")


def generate_openwebtext():
    ds = load_dataset("Skylion007/openwebtext", split="train", streaming=True)
    n = 300_000
    ds = get_sample_records(ds, n)
    write_sample_records(ds, f"src/resources/openwebtext_sample_{n}.bin")


# generate_tinystores()
generate_openwebtext()