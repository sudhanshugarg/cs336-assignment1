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

PST = ZoneInfo("America/Los_Angeles")
params = {
    "vocab_size": 32000,
    "token_dim": 64,
    "endecoder_layers": 2,
    "max_seq_length": 5000,
    "n_heads": 4,
    "d_ff": 4,
    "theta": 10000,
    "dtype": "float16",
    "device": "mps",
    "batch_size": 16
}
tokenizer_file_path = f"src/resources/tinystories_vocab_{params['vocab_size']}_50.pkl"

wandb_log = True

class TransformerCrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        #preds.shape = [batch, seq_length, vocab_size]
        #labels.shape = [batch, seq_length]

        # logits = logits.reshape(-1, logits.shape[-1])
        # labels = labels.reshape(-1)

        #lets make sure issues due to large or tiny logits is resolved.
        max_logits_per_row = torch.max(logits, dim=-1, keepdim=True).values

        # print("logits.shape: ", logits)
        # print("labels.shape: ", labels)
        # print("max logits: ", max_logits_per_row)

        log_logits_exp_sum = torch.log(torch.sum(torch.exp(logits - max_logits_per_row), dim=-1, keepdim=True))
        # print(log_logits_exp_sum)

        # relevant_label_logits = logits[torch.arange(logits.shape[-2]), labels]
        logits = logits.reshape(-1, logits.shape[-1]) # n x d
        labels = labels.reshape(-1) #n x 1
        relevant_label_logits = logits[torch.arange(logits.shape[0]), labels]
        relevant_label_logits = relevant_label_logits.reshape(max_logits_per_row.shape)
        # print("relevant_label_logits: ", relevant_label_logits)
        return torch.mean(max_logits_per_row + log_logits_exp_sum - relevant_label_logits)


def get_tensor(tokenizer: Tokenizer_V1, x: list[str]) -> torch.Tensor:
    tokens = tokenizer.tokenize(x, seq_length=seq_length)
    token_ints = []
    for token_str, token_int in tokens:
        token_ints.append(token_int)
    return torch.tensor(token_ints)


def train_step(it: int, model: nn.Module, x: torch.Tensor, y: torch.Tensor, seq_length: int, device = None, print_every: int = 10):
    model.train()
    logits, probs = model(x) #b, seq, vocab_size
    #each of the 3 indexes are broadcast into y.shape
    #mask for y also has to be applied.
    #we only take losses from y, for non-padded positions.
    eps = 1e-8

    #y.shape = b, seq_length
    print("y.shape = ", y.shape)
    print("probs.shape = ", probs.shape)
    print("logits.shape = ", logits.shape)
    label_mask = ((y != Tokenizer_V1.padding_token_int) * 1).to(device)
    # print(label_mask)
    losses = probs[torch.arange(logits.shape[0])[:, None], torch.arange(logits.shape[1])[None, :], y].to(device)
    # print(probs[:3, :4])
    losses = -torch.log(losses + eps) * label_mask
    print("sum loss = ", losses.sum())
    loss = losses.sum() / (label_mask.sum() + eps)

    if it % print_every == 0:
        print(f"{it}: loss = {loss.item()}")
        eval_model(model, seq_length=seq_length, device=device)

    if wandb_log:
        wandb.log({
            "iter": it,
            "loss": loss.item()
        })
    loss.backward()


def train(max_steps: int, seq_length: int, device=None):
    assert seq_length <= params["max_seq_length"]

    if wandb_log:
        wandb.init(project="sudgarg", name="xformer_scratch")
    train_start = datetime.now(PST)
    train_start_str = train_start.strftime("%s")

    torch.manual_seed(157)
    model = Transformer(**params)
    tokenizer = Tokenizer.from_files(
        vocab_filepath=tokenizer_file_path,
        merges_filepath=tokenizer_file_path,
        special_tokens=["<|endoftext|>"])
    # tokenizer.tokenize()

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

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
        batch_y = batch_y.to(device).long()
        train_step(i, model, batch_x, batch_y, seq_length=seq_length, device=device, print_every=3)
        optimizer.step()

    
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

    print(start, end="")
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
    seq_length = 8
    device = torch.device(params["device"])
    ckpt_path = train(500, seq_length=seq_length, device=device)
    eval(model_path=ckpt_path, seq_length=seq_length, device=device)
