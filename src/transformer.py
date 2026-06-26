from typing import Any

import torch
import torch.nn as nn
import torch.nn.init as init
import math

"""
input is [batch, seq_length] -> of integers (tokenIds)
1. convert tokenIds -> embeddings -> [batch, seq_length, token_dim] (DONE)
2. pre layer_norm (DONE)

ENCODER/DECODER BLOCK X 4
2.1 positional encoding (todo)
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
        attention_params = kwargs.copy()
        attention_params.update({
            "n_heads": 2
        })
        self.attention = CausalSelfAttention(**attention_params)
        self.ln = LayerNorm()

        mlp_params = kwargs.copy()
        mlp_params.update({
            "hidden_layer_dim": 4,
            "hidden_layers": 2
        })
        self.mlp = MLP(**mlp_params)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.attention(x, True)
        x = self.ln(x)
        return self.mlp(x)

class HomeReLU(nn.Module):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mask = (x > 0).to(torch.int16)
        return x * mask


class Linear(nn.Module):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self.input_dim = kwargs.pop("input_dim")
        self.output_dim = kwargs.pop("output_dim")
        self.layer = nn.Parameter(torch.empty(self.input_dim, self.output_dim))
        self.init_layer()
    
    def init_layer(self) -> None:
        dim0, dim1 = self.layer.shape
        sigma = 2.0 / (dim0 + dim1)
        init.trunc_normal_(self.layer, mean=0.0, std=sigma, a=-3.0*sigma, b=3.0*sigma)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # b, seq, token_dim = x.shape
        return x @ self.layer


class MLP(nn.Module):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self.token_dim = kwargs["token_dim"]
        self.hidden_layers = kwargs.pop("hidden_layers")
        self.hidden_layer_dim = kwargs.pop("hidden_layer_dim")

        self.layers = []
        if self.hidden_layers <= 0:
            layer = Linear(**{"input_dim": self.token_dim, "output_dim": self.token_dim})
            # layer = nn.Parameter(torch.empty(self.token_dim, self.token_dim))
            # self.init_layer(layer)
            self.layers.append(layer)
        else:
            # layer = nn.Parameter(torch.empty(self.token_dim, self.hidden_layer_dim))
            layer = Linear(**{"input_dim": self.token_dim, "output_dim": self.hidden_layer_dim})
            # self.init_layer(layer)
            self.layers.append(layer)
            self.layers.append(HomeReLU())

            for i in range(self.hidden_layers-1):
                # layer = nn.Parameter(torch.empty(self.hidden_layer_dim, self.hidden_layer_dim))
                # self.init_layer(layer)
                layer = Linear(**{"input_dim": self.hidden_layer_dim, "output_dim": self.hidden_layer_dim})
                self.layers.append(layer)
                self.layers.append(HomeReLU())
            # layer = nn.Parameter(torch.empty(self.hidden_layer_dim, self.token_dim))
            # self.init_layer(layer)
            layer = Linear(**{"input_dim": self.hidden_layer_dim, "output_dim": self.token_dim})
            self.layers.append(layer)


    def init_layer(self, layer: nn.Parameter) -> None:
        dim0, dim1 = layer.shape
        sigma = 2.0 / (dim0 + dim1)
        init.trunc_normal_(layer, mean=0.0, std=sigma, a=-3.0*sigma, b=3.0*sigma)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i in range(len(self.layers)):
            x = self.layers[i](x)
        return x

class CausalSelfAttention(nn.Module):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        """
        create qkv, up_proj
        """
        self.token_dim = kwargs["token_dim"]
        self.n_heads = kwargs.pop("n_heads")
        assert(self.n_heads > 0 and self.token_dim % self.n_heads == 0)
        self.head_dim = self.token_dim // self.n_heads

        self.Q = nn.Parameter(torch.empty(self.token_dim, self.token_dim))
        self.K = nn.Parameter(torch.empty(self.token_dim, self.token_dim))
        self.V = nn.Parameter(torch.empty(self.token_dim, self.token_dim))
        self.up_proj = nn.Parameter(torch.empty(self.token_dim, self.token_dim))

        sigma = 2.0 / (2.0 * math.sqrt(self.token_dim))
        init.trunc_normal_(self.Q, mean=0.0, std=sigma, a=-3.0 * sigma, b=3.0 * sigma)
        init.trunc_normal_(self.K, mean=0.0, std=sigma, a=-3.0 * sigma, b=3.0 * sigma)
        init.trunc_normal_(self.V, mean=0.0, std=sigma, a=-3.0 * sigma, b=3.0 * sigma)
        init.trunc_normal_(self.up_proj, mean=0.0, std=sigma, a=-3.0 * sigma, b=3.0 * sigma)


    def _upper_triangular(self, n: int) -> torch.Tensor:
        rows = torch.arange(n).view(n, 1)
        cols = torch.arange(n).view(-1, n)
        return rows < cols

    def _stable_softmax(self, x: torch.Tensor) -> torch.Tensor:
        max_logit = torch.max(x, dim=-1, keepdim=True).values
        x = torch.exp(x - max_logit) #b, n_heads, seq, seq
        sum = torch.sum(x, dim=-1, keepdim=True)
        return x / sum


    def forward(self, x: torch.Tensor, do_mask: bool) -> torch.Tensor:
        b, seq, tok = x.shape
        assert(tok == self.token_dim)

        q = torch.matmul(x, self.Q) #b, seq, tok_dim
        q = q.reshape(b, seq, self.n_heads, self.head_dim).transpose(1, 2) #b, n_heads, seq, head_dim
        k = torch.matmul(x, self.Q) #b, seq, tok_dim
        k = k.reshape(b, seq, self.n_heads, self.head_dim).transpose(1, 2) #b, n_heads, seq, head_dim
        v = torch.matmul(x, self.V)
        v = v.reshape(b, seq, self.n_heads, self.head_dim).transpose(1, 2) #b, n_heads, seq, head_dim

        attention = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim) #b, n_heads, seq, seq
        if do_mask:
            mask = self._upper_triangular(seq)
            attention = attention.masked_fill(mask, float("-inf"))

        attention_softmax = self._stable_softmax(attention) #b, n_heads, seq, seq

        result = torch.matmul(attention_softmax, v) #b, n_heads, seq, head_dim
        result = result.transpose(1, 2).reshape(b, seq, self.token_dim)
        return torch.matmul(result, self.up_proj)


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

        self.vocab_size = kwargs.pop("vocab_size")
        self.token_dim = kwargs["token_dim"]
        self.endecoder_layers = kwargs.pop("endecoder_layers")

        self.tokenEmbeddings = nn.Parameter(torch.empty(self.vocab_size, self.token_dim))
        init.trunc_normal_(self.tokenEmbeddings, mean=0.0, std=1.0, a=-3.0, b=3.0)
        self.ln = LayerNorm()

        # with torch.no_grad():
        #     self.tokenEmbeddings.uniform_(0.0, 0.02)    
        # print(self.tokenEmbeddings)
        self.layers = nn.ModuleList()
        for i in range(self.endecoder_layers):
            self.layers.append(EnDecoder(**kwargs))


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = x.shape

        input_tensor = self.tokenEmbeddings[x]
        pre_ln_input = self.ln(input_tensor)
        # print(pre_ln_input.shape)
        # print(pre_ln_input)

        # print(torch.std(pre_ln_input, dim=-1))

        y = pre_ln_input
        for i in range(self.endecoder_layers):
            y = self.layers[i](y)

        print(x.shape)
        print(y.shape)
        return y
        

        

torch.manual_seed(157)
params = {
    "vocab_size": 6,
    "token_dim": 4,
    "endecoder_layers": 2
}
t = Transformer(**params)
input = torch.tensor([
    [0, 1],
    [1, 2],
    [0, 3],
])

print(t(input))