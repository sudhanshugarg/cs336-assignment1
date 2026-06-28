import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformer import Transformer
from tokenizer import Tokenizer
from dataset import TextFileReader


def get_tensor(tokenizer: Tokenizer, x: list[str]) -> torch.Tensor:
    tokens = tokenizer.tokenize(x)
    token_ints = []
    for token_str, token_int in tokens:
        token_ints.append(token_int)
    return torch.tensor(token_ints)


def train_step(model: nn.Module, x: torch.Tensor, y: torch.Tensor):
    model.train()
    logits, probs = model(x)
    losses = logits[y] #cross entropy loss
    losses = torch.mean(-torch.log(losses))


def train():
    params = {
        "vocab_size": 100,
        "token_dim": 4,
        "endecoder_layers": 2
    }
    torch.manual_seed(157)
    model = Transformer(**params)
    file_name = "input.txt"
    file_path = f"src/resources/{file_name}"
    tokenizer_path = f"src/resources/tokenizer_{file_name}_{params['vocab_size']}.pkl"
    tokenizer = Tokenizer(file_path, tokenizer_path)
    # tokenizer.tokenize()

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
    optimizer.zero_grad(set_to_none=True)

    seq_length = 6
    dataset = TextFileReader(file_path, seq_length=seq_length)
    dataloader = DataLoader(dataset=dataset, batch_size=3, shuffle=False)

    max_steps = 1
    data_iter = iter(dataloader)
    for _ in range(max_steps):
        try:
            batch_x, batch_y = next(data_iter)    
        except StopIteration as e:
            print(f"looping over batches since {e}")
            data_iter = iter(dataloader)
            batch_x, batch_y = next(data_iter)


        x = get_tensor(tokenizer, batch_x)
        y = get_tensor(tokenizer, batch_y)
        train_step(model, x, y)


train()

