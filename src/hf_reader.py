from datasets import load_dataset, Dataset, load_from_disk

def get_sample_records(ds, n: int = 0):
    # records = ds.select(range(n))
    print(ds.column_names)
    # print(len(ds))
    if n > 0:
        records = ds.take(n)
    else:
        records = ds
    return records

def write_sample_records(ds, path):
    endToken = "<|endoftext|>"
    with open(path, "w") as f:
        for row in ds:
            text = row["text"].strip()
            f.write(f"{text}{endToken}\n")

def generate_tinystores(n: int = 0):
    ds = load_dataset("roneneldan/TinyStories", split="train", streaming=False)
    ds = get_sample_records(ds, n)
    write_sample_records(ds, f"src/resources/tinystories_sample_{n}.bin")


def generate_openwebtext(n: int = 0):
    ds = load_dataset("Skylion007/openwebtext", split="train", streaming=False)
    ds = get_sample_records(ds, n)
    write_sample_records(ds, f"src/resources/openwebtext_sample_{n}.bin")


# generate_tinystores()
generate_openwebtext()