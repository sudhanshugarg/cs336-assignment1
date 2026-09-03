import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb
from datetime import datetime
from zoneinfo import ZoneInfo

from src.transformer import Utils
from src.transformer import Transformer
from src.tokenizer_v1 import Tokenizer_V1
from src.tokenizer import Tokenizer
from src.dataset import TextFileReader
from src.transformer import compute_params_and_flops
from src.loss import TransformerCrossEntropyLoss, CrossEntropyFromProbabilities, CrossEntropyFromLogits

PST = ZoneInfo("America/Los_Angeles")
params = {
    "vocab_size": 32000,
    "token_dim": 768,
    "endecoder_layers": 4,
    "max_seq_length": 2048,
    "n_heads": 1,
    "d_ff": 4,
    "theta": 10000,
    "dtype": "float32",
    "device": "mps",
    "batch_size": 32
}
tokenizer_file_path = f"src/resources/tinystories_vocab_{params['vocab_size']}_50.pkl"

wandb_log = False



def get_tensor(tokenizer: Tokenizer_V1, x: list[str]) -> torch.Tensor:
    tokens = tokenizer.tokenize(x, seq_length=seq_length)
    token_ints = []
    for token_str, token_int in tokens:
        token_ints.append(token_int)
    return torch.tensor(token_ints)


def train_step(
        it: int, 
        model: nn.Module, 
        lossFn: nn.Module, 
        x: torch.Tensor, 
        y: torch.Tensor, 
        seq_length: int, 
        device=None, 
        print_every: int = 10
    ):

    model.train()
    logits, probs = model(x) #b, seq, vocab_size
    top_k = 5

    top_k_probs = probs[:, :, :7]
    loss = lossFn(logits, y)

    if it % print_every == 0:
        print(f"{it}: loss = {loss.item()}")
        # print("x = ", x, ", y = ", y)
        # print("top_k_probs =", top_k_probs)
        eval_model(model, seq_length=seq_length, device=device)

    if wandb_log:
        wandb.log({
            "iter": it,
            "loss": loss.item()
        })
    return loss


def train(
        max_steps: int, 
        seq_length: int, 
        device=None, 
        print_every: int = 10, 
        save_every: int = 50, 
        checkpoint=None
    ):
    assert seq_length <= params["max_seq_length"]

    if wandb_log:
        wandb.init(project="sudgarg", name="xformer_scratch")
    train_start = datetime.now(PST)
    train_start_str = train_start.strftime("%s")

    torch.manual_seed(157)
    model = Transformer(**params)
    if checkpoint:
        model.load_state_dict(checkpoint)

    # lossFn = CrossEntropyFromProbabilities(device=device)
    lossFn = TransformerCrossEntropyLoss()
    tokenizer = Tokenizer.from_files(
        vocab_filepath=tokenizer_file_path,
        merges_filepath=tokenizer_file_path,
        special_tokens=["<|endoftext|>"])
    # tokenizer.tokenize()

    # optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,          # try 1e-4 or 5e-5 if unstable
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.1,
    )
    file_path = f"src/resources/input.txt"

    dataset = TextFileReader(file_path, seq_length=seq_length, tokenizer=tokenizer)
    dataloader = DataLoader(dataset=dataset, batch_size=params["batch_size"], shuffle=True)

    data_iter = iter(dataloader)
    for i in range(max_steps):
        try:
            batch_x, batch_y = next(data_iter)    
        except StopIteration as e:
            print(f"looping over batches since {e}")
            data_iter = iter(dataloader)
            batch_x, batch_y = next(data_iter)

        # x = get_tensor(tokenizer, batch_x)
        # y = get_tensor(tokenizer, batch_y)
        optimizer.zero_grad(set_to_none=True)
        batch_x = batch_x.to(device)
        # print("batch_x =", batch_x)
        batch_y = batch_y.to(device).long()
        # print("batch_y =", batch_y)
        loss = train_step(i, model, lossFn, batch_x, batch_y, seq_length=seq_length, device=device, print_every=print_every)
        loss.backward()
        optimizer.step()

        if i % save_every == 0:
            model_path = f"src/resources/{train_start_str}_{i}_checkpoint.pt"
            torch.save(model.state_dict(), model_path)
            print(f"saved checkpoint {model_path}")

    
    model_path = f"src/resources/{train_start_str}-checkpoint.pt"
    torch.save(model.state_dict(), model_path)
    print(f"saved checkpoint {model_path}")
    return model_path

def bucketize(query_1d: torch.Tensor, boundaries_2d: torch.Tensor) -> torch.Tensor:
    # query is of size bx1, boundaries of size b x vocab
    # result will be bx1, one index from boundaries for each

    m = (query_1d <= boundaries_2d) * 1 #b x vocab
    return boundaries_2d.shape[-1] - torch.sum(m, dim=-1)


def sample_probs(probs: torch.Tensor, device=None) -> torch.Tensor:
    #probs.shape = [b, seq_length, vocab_size]
    #b = 1
    cumprobs = torch.cumsum(probs[:, -1], dim=-1)
    cumprobs[..., -1] = 1.0
    # print(probs[:, -1].shape)
    # print(cumprobs.shape)
    return bucketize(torch.rand(probs.shape[0], device=device), cumprobs)

def eval(model_path: str, seq_length: int, device=None):
    torch.manual_seed(157)
    if not os.path.exists(model_path):
        raise ValueError(f"{model_path} not found")

    checkpoint = torch.load(model_path)


    model = Transformer(**params)
    model.load_state_dict(checkpoint)
    eval_model(model, seq_length, device)

def eval_model(model: nn.Module, seq_length: int, device=None):
    assert seq_length <= params["max_seq_length"]
    model.eval()

    #start string
    #tokenize it
    #get predictions
    #enter the same string again
    #get next predictions
    #continue forever
    start = "Julius how goes it. it has been a while since you met cleo. All good "
    tokenizer = Tokenizer.from_files(
        vocab_filepath=tokenizer_file_path,
        merges_filepath=tokenizer_file_path,
        special_tokens=["<|endoftext|>"]
    )
    max_length = 50
    token_ints = tokenizer.encode(start)

    with torch.no_grad():
        for i in range(max_length):
            x = torch.tensor(token_ints[-seq_length:], device=device).unsqueeze(dim=0)
            # print(x.shape)
            # break
            _, probs = model(x)
            # print(probs.shape)
            #need to sample from these probabilities
            batch_next_token_ints = sample_probs(probs, device=device)
            next_token_int = int(batch_next_token_ints[0].item())
            token_ints.append(next_token_int)
        print(tokenizer.decode(token_ints), end="\n")
    print()


if __name__ == "__main__":
    param_count, flops = compute_params_and_flops(
        layers=params["endecoder_layers"],
        b=params["batch_size"],
        s=params["max_seq_length"],
        t=params["token_dim"],
        v=params["vocab_size"],
        d_ff=params["d_ff"],
        n=params["n_heads"]
    )
    print(f"for this training run, param_count = {param_count}, estimated flops per forward pass: {flops}")
    seq_length = 1024
    device = torch.device(params["device"])
    # checkpoint = torch.load("src/resources/1788436876-checkpoint.pt")
    checkpoint = None
    ckpt_path = train(max_steps=50000, print_every=10, save_every=100, seq_length=seq_length, device=device, checkpoint=checkpoint)
    eval(model_path=ckpt_path, seq_length=seq_length, device=device)
