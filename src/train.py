import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformer import Transformer
from tokenizer import Tokenizer
from dataset import TextFileReader
import wandb
from datetime import datetime
from zoneinfo import ZoneInfo
import os

PST = ZoneInfo("America/Los_Angeles")
params = {
    "vocab_size": 1000,
    "token_dim": 16,
    "endecoder_layers": 2,
    "n_heads": 2,
    "mlp_hidden_layer_dim": 4,
    "mlp_hidden_layers": 2
}
seq_length = 32


def get_tensor(tokenizer: Tokenizer, x: list[str]) -> torch.Tensor:
    tokens = tokenizer.tokenize(x)
    token_ints = []
    for token_str, token_int in tokens:
        token_ints.append(token_int)
    return torch.tensor(token_ints)


def train_step(it: int, model: nn.Module, x: torch.Tensor, y: torch.Tensor):
    model.train()
    logits, probs = model(x)
    #each of the 3 indexes are broadcast into y.shape
    losses = probs[torch.arange(logits.shape[0])[:, None], torch.arange(logits.shape[1])[None, :], y]
    # print(losses)
    loss = losses.mean()
    
    if it % 500 == 0:
        print(f"{it}: loss = {loss.item()}")
    wandb.log({
        "iter": it,
        "loss": loss.item()
    })
    loss.backward()


def train(max_steps: int):

    wandb.init(project="sudgarg", name="xformer_scratch")
    train_start = datetime.now(PST)
    train_start_str = train_start.strftime("%s")

    torch.manual_seed(157)
    model = Transformer(**params)
    file_name = "input.txt"
    file_path = f"src/resources/{file_name}"
    tokenizer_path = f"src/resources/tokenizer_{file_name}_{params['vocab_size']}.pkl"
    tokenizer = Tokenizer(file_path, tokenizer_path, vocab_size=params["vocab_size"], overwrite=True)
    # tokenizer.tokenize()

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    dataset = TextFileReader(file_path, seq_length=seq_length, tokenizer=tokenizer)
    dataloader = DataLoader(dataset=dataset, batch_size=16, shuffle=True)

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
        train_step(i, model, batch_x, batch_y)
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


def sample_probs(probs: torch.Tensor) -> torch.Tensor:
    #probs.shape = [b, seq_length, vocab_size]
    #b = 1
    cumprobs = torch.cumsum(probs[:, -1], dim=-1)
    # print(probs[:, -1].shape)
    # print(cumprobs.shape)
    return bucketize(torch.rand(probs.shape[0]), cumprobs)

def eval(model_path: str):
    torch.manual_seed(157)
    if not os.path.exists(model_path):
        raise ValueError(f"{model_path} not found")

    checkpoint = torch.load(model_path)


    model = Transformer(**params)
    model.load_state_dict(checkpoint)
    model.eval()

    #start string
    #tokenize it
    #get predictions
    #enter the same string again
    #get next predictions
    #continue forever
    start = "Julius how goes it these days. I'm good cleopatra, nothhing new to report"
    tokenizer_path = f"src/resources/tokenizer_input.txt_{params['vocab_size']}.pkl"
    tokenizer = Tokenizer(corpus_file_path="", tokenizer_path=tokenizer_path, vocab_size=params["vocab_size"])

    max_length = 50
    tokens, token_ints = tokenizer.tokenize([start])[0]

    print(start, end="")
    with torch.no_grad():
        for i in range(max_length):
            x = torch.tensor(token_ints[-seq_length:]).unsqueeze(dim=0)
            # print(x.shape)
            # break
            _, probs = model(x)
            # print(probs.shape)
            #need to sample from these probabilities
            batch_next_token_ints = sample_probs(probs)
            next_token_int = int(batch_next_token_ints[0].item())
            token_ints.append(next_token_int)
            print(tokenizer.tokenMapInt[next_token_int], end="")
    print()

ckpt_path = train(3000)
eval(model_path=ckpt_path)
