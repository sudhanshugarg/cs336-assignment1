from typing import Any

import torch
import torch.nn as nn
import torch.nn.init as init
"""
input is [batch, seq_length] -> of integers (tokenIds)
1. convert tokenIds -> embeddings -> [batch, seq_length, token_dim] (DONE)
2. pre layer_norm

ENCODER/DECODER BLOCK X 4

3. causalselfattention (multi headed self attention)
4. layer norm
5. mlp
6. layer norm

(residual ?)

OUTPUT LAYER
top_k (top k tokens), or top_p (cumulative probability reaches p)
[b, seq, token_dim] -> logits (softmax)
train: cross entropy loss against label of next token
eval: sample from logits and take next token, and
continue

"""


class EnDecoder(nn.Module):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass


class MLP(nn.Module):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass


class CausalSelfAttention(nn.Module):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass


class LayerNorm(nn.Module):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()

        self.eps = 1e-8
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mu = torch.mean(x, dim=-1, keepdim=True) #column
        sigma = torch.std(x, dim=-1, keepdim=True)
        return (x - mu) / (sigma + self.eps)



class Transformer(nn.Module):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()

        vocab_size = kwargs.pop("vocab_size")
        token_dim = kwargs.pop("token_dim")
        self.tokenEmbeddings = nn.Parameter(torch.empty(vocab_size, token_dim))
        init.trunc_normal_(self.tokenEmbeddings, mean=0.0, std=1.0, a=-3.0, b=3.0)
        self.ln = LayerNorm()

        # with torch.no_grad():
        #     self.tokenEmbeddings.uniform_(0.0, 0.02)
    
        print(self.tokenEmbeddings)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = x.shape

        input_tensor = self.tokenEmbeddings[x]
        pre_ln_input = self.ln(input_tensor)
        print(pre_ln_input.shape)
        print(pre_ln_input)

        print(torch.std(pre_ln_input, dim=-1))

        return x
        

        

torch.manual_seed(157)
params = {
    "vocab_size": 6,
    "token_dim": 4
}
t = Transformer(**params)
input = torch.tensor([
    [0, 1],
    [1, 2],
    [0, 3],
])

t(input)