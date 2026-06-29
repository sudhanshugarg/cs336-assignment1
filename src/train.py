import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformer import Transformer
from tokenizer import Tokenizer
from dataset import TextFileReader
import wandb


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


def train():
    params = {
        "vocab_size": 1000,
        "token_dim": 16,
        "endecoder_layers": 2,
        "n_heads": 2,
        "mlp_hidden_layer_dim": 4,
        "mlp_hidden_layers": 2
    }
    wandb.init(project="sudgarg", name="xformer_scratch")
    torch.manual_seed(157)
    model = Transformer(**params)
    file_name = "input.txt"
    file_path = f"src/resources/{file_name}"
    tokenizer_path = f"src/resources/tokenizer_{file_name}_{params['vocab_size']}.pkl"
    tokenizer = Tokenizer(file_path, tokenizer_path, vocab_size=params["vocab_size"], overwrite=True)
    # tokenizer.tokenize()

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    seq_length = 32
    dataset = TextFileReader(file_path, seq_length=seq_length, tokenizer=tokenizer)
    dataloader = DataLoader(dataset=dataset, batch_size=16, shuffle=True)

    max_steps = 3500
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


train()

