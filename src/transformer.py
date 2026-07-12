from typing import Any

import torch
import torch.nn as nn
import torch.nn.init as init
import math

"""
input is [batch, seq_length] -> of integers (tokenIds) [TODO fix the current implementation of tokenizer]

1. convert tokenIds -> embeddings -> [batch, seq_length, token_dim] (DONE)
2. pre layer_norm (DONE)
3. TODO padding when seq_length < context_length
4. (DONE) take the correct tokens, earlier is better.

ENCODER/DECODER BLOCK X 4
2.1 positional encoding - cosine/rope (TODO)
3. causalselfattention (multi headed self attention) [TODO] masking and padding
4. layer norm [DONE] scale and shift params
5. mlp [DONE]
6. MOE [TODO]
7. layer norm [DONE] scale and shift params

(residual DONE)

OUTPUT LAYER
top_k (top k tokens), or top_p (cumulative probability reaches p)
[b, seq, token_dim] -> logits (softmax) [DONE]


TRAINING
optimizer - write by self TODO
train: cross entropy loss against label of next token
wandb monitor TODO

EVAL
eval: sample from logits and take next token [DONE]
"""
class Utils():
    @staticmethod
    def stable_softmax(x: torch.Tensor, dimension: int=-1) -> torch.Tensor:
        max_logit = torch.max(x, dim=dimension, keepdim=True).values
        x = torch.exp(x - max_logit) #b, n_heads, seq, seq
        sum = torch.sum(x, dim=dimension, keepdim=True)
        return x / sum

class EnDecoder(nn.Module):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        attention_params = kwargs.copy()
        self.attention = CausalSelfAttention(**attention_params)
        mlp_params = kwargs.copy()
        self.mlp = MLP(**mlp_params)
        self.ln = LayerNorm(dim=kwargs["token_dim"])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #PreLN norm residual
        # x1 = x.clone()
        # x1 = self.ln(x1)
        # x1 = self.attention(x1, True)
        # x1 = x + x1
        x = x + self.attention(self.ln(x), use_upper_triangular=True)

        # x2 = x1.clone()
        # x2 = self.ln(x2)
        # x2 = self.mlp(x2)
        x = x + self.mlp(self.ln(x))
        return x

class HomeReLU(nn.Module):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mask = (x > 0).to(torch.int16)
        return x * mask

class SiLU(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)

class FFN(nn.Module):
    def __init__(self, d_ff: int, d_model: int) -> None:
        super().__init__()
        self.d_ff = d_ff
        self.d_model = d_model
        self.w1 = Linear2(in_features=d_model, out_features=d_ff)
        self.w2 = Linear2(in_features=d_ff, out_features=d_model)
        self.w3 = Linear2(in_features=d_model, out_features=d_ff)
        self.silu = SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[-1] == self.d_model
        w1_x = self.w1(x) # ..., d_ff
        w1_x = self.silu(w1_x)
        w3_x = self.w3(x) # ..., d_ff
        w2_x = self.w2(w1_x * w3_x)
        return w2_x

class Linear2(nn.Module):
    def __init__(self, in_features: int, out_features: int, device = None, dtype = None) -> None:
        super().__init__()
        self.input_dim = in_features
        self.output_dim = out_features
        self.layerAAA = nn.Parameter(torch.empty(self.input_dim, self.output_dim, device=device, dtype=dtype))
        self.init_layer()

    def init_layer(self) -> None:
        dim0, dim1 = self.layerAAA.shape
        sigma = 2.0 / (dim0 + dim1)
        init.trunc_normal_(self.layerAAA, mean=0.0, std=sigma, a=-3.0*sigma, b=3.0*sigma)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # b, seq, token_dim = x.shape
        return x @ self.layerAAA


@DeprecationWarning
class Linear(nn.Module):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self.input_dim = kwargs.pop("input_dim")
        self.output_dim = kwargs.pop("output_dim")
        if "weights" not in kwargs:
            self.layer = nn.Parameter(torch.empty(self.input_dim, self.output_dim))
            self.init_layer()
        else:
            weights = kwargs.pop("weights")
            self.layer = nn.Parameter(weights)
    
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
        self.hidden_layers = kwargs["mlp_hidden_layers"]
        self.hidden_layer_dim = kwargs["mlp_hidden_layer_dim"]

        self.layers = []
        if self.hidden_layers <= 0:
            # layer = Linear(**{"input_dim": self.token_dim, "output_dim": self.token_dim})
            layer = Linear2(in_features=self.token_dim, out_features=self.token_dim)
            # layer = nn.Parameter(torch.empty(self.token_dim, self.token_dim))
            # self.init_layer(layer)
            self.layers.append(layer)
        else:
            # layer = nn.Parameter(torch.empty(self.token_dim, self.hidden_layer_dim))
            # layer = Linear(**{"input_dim": self.token_dim, "output_dim": self.hidden_layer_dim})
            layer = Linear2(in_features=self.token_dim, out_features=self.hidden_layer_dim)
            # self.init_layer(layer)
            self.layers.append(layer)
            self.layers.append(HomeReLU())

            for i in range(self.hidden_layers-1):
                # layer = nn.Parameter(torch.empty(self.hidden_layer_dim, self.hidden_layer_dim))
                # self.init_layer(layer)
                # layer = Linear(**{"input_dim": self.hidden_layer_dim, "output_dim": self.hidden_layer_dim})
                layer = Linear2(in_features=self.hidden_layer_dim, out_features=self.hidden_layer_dim)
                self.layers.append(layer)
                self.layers.append(HomeReLU())
            # layer = nn.Parameter(torch.empty(self.hidden_layer_dim, self.token_dim))
            # self.init_layer(layer)
            # layer = Linear(**{"input_dim": self.hidden_layer_dim, "output_dim": self.token_dim})
            layer = Linear2(in_features=self.hidden_layer_dim, out_features=self.token_dim)
            self.layers.append(layer)

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
        self.n_heads = kwargs["n_heads"]
        assert(self.n_heads > 0 and self.token_dim % self.n_heads == 0)
        self.head_dim = self.token_dim // self.n_heads

        self.Q = Linear2(self.token_dim, self.token_dim)
        self.K = Linear2(self.token_dim, self.token_dim)
        self.V = Linear2(self.token_dim, self.token_dim)
        self.up_proj = Linear2(self.token_dim, self.token_dim)

        max_seq_length = kwargs["seq_length"]
        device = kwargs["device"]
        self.rope = RotaryPositionalEmbedding(theta=10000.0, d_k=self.token_dim, max_seq_length=max_seq_length, device=device)


    def _upper_triangular(self, n: int) -> torch.Tensor:
        rows = torch.arange(n).view(n, 1)
        cols = torch.arange(n).view(-1, n)
        # we want 1s to represent the values to keep, and 0s to represent
        # the values to set to -inf. hence return negation.
        return ~(rows < cols)

    @staticmethod
    def scaled_dot_product_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        #q.shape = ..., b, n_heads, seq, head_dim
        attention = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1]) #b, n_heads, seq, seq
        if mask is not None:
            attention = attention.masked_fill(~mask, float("-inf"))

        attention_softmax = Utils.stable_softmax(attention) #b, n_heads, seq, seq
        result = torch.matmul(attention_softmax, v) #b, n_heads, seq, head_dim
        return result

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None, use_upper_triangular: bool = False) -> torch.Tensor:
        b, seq, tok = x.shape
        assert(tok == self.token_dim)

        token_positions = torch.arange(seq)

        q = self.Q(x) #b, seq, tok_dim
        q = self.rope(q, token_positions)
        q = q.reshape(b, seq, self.n_heads, self.head_dim).transpose(1, 2) #b, n_heads, seq, head_dim
        k = self.K(x) #b, seq, tok_dim
        k = self.rope(k, token_positions)
        k = k.reshape(b, seq, self.n_heads, self.head_dim).transpose(1, 2) #b, n_heads, seq, head_dim
        v = self.V(x)
        v = v.reshape(b, seq, self.n_heads, self.head_dim).transpose(1, 2) #b, n_heads, seq, head_dim

        # attention = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim) #b, n_heads, seq, seq
        # if mask:
        #     attention = attention.masked_fill(mask, float("-inf"))
        # elif use_upper_triangular:
        #     mask = self._upper_triangular(seq)
        #     attention = attention.masked_fill(mask, float("-inf"))

        # attention_softmax = Utils.stable_softmax(attention) #b, n_heads, seq, seq
        # result = torch.matmul(attention_softmax, v) #b, n_heads, seq, head_dim

        if mask is None and use_upper_triangular:
            mask = self._upper_triangular(seq)
        scaled_dot_prod_attn = self.scaled_dot_product_attention(q, k, v, mask)
        result = scaled_dot_prod_attn.transpose(1, 2).reshape(b, seq, self.token_dim)
        return self.up_proj(result)


class LayerNorm(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.gamma = nn.Parameter(torch.empty(self.dim))
        self.beta = nn.Parameter(torch.empty(self.dim))
        sigma = (1.0 / math.sqrt(self.dim))
        init.trunc_normal_(self.gamma, mean=0.0, std=sigma, a=-3.0*sigma, b=3.0*sigma)
        init.trunc_normal_(self.beta, mean=0.0, std=sigma, a=-3.0*sigma, b=3.0*sigma)
        self.eps = 1e-8
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #x = [b, seq, token_dim]
        mu = torch.mean(x, dim=-1, keepdim=True) #column
        sigma = torch.std(x, dim=-1, keepdim=True)
        return (((x - mu) / (sigma + self.eps)) * self.gamma) + self.beta

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.empty(d_model, device=device, dtype=dtype))
        init.trunc_normal_(self.gamma, mean=0, std=1.0, a=-2.0, b=2.0)
        self.d_model = d_model
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[-1] == self.d_model
        in_dtype = x.dtype
        x = x.to(torch.float32)
        #what is rms norm
        denominator = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return ((x / denominator) * self.gamma).to(in_dtype)

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_length: int, device=None):
        super().__init__()
        assert d_k % 2 == 0
        self.d_k = d_k
        self.d_k_2 = d_k // 2
        self.theta = theta
        self.max_seq_length = max_seq_length
        self.device = device

        # for pre-computing the angles for all positions
        powers = -2.0 * torch.arange(self.d_k // 2, dtype=torch.float32, device=device) / self.d_k
        thetas = torch.pow(theta, powers) #1xd/2
        # print(token_positions.unsqueeze(dim=-1).shape)
        angles = torch.arange(max_seq_length, device=device, dtype=torch.float32).unsqueeze(dim=-1) * thetas[None, :] #[..., seq_len, d/2]
        # print("angles: ", angles.shape)
        cos = torch.cos(angles) #...,seq_len,d/2
        # print("cos_angles: ", cos_angles.shape)
        sin = torch.sin(angles) #...,seq_len,d/2

        self.register_buffer("cos_angles", cos)
        self.register_buffer("sin_angles", sin)

    def _using_block_diag(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        seq_length, token_dim = x.shape[-2:]
        assert token_dim == self.d_k

        print("x: ", x.shape)
        powers = -2.0 * torch.arange(self.d_k // 2, dtype=torch.float32) / self.d_k
        thetas = torch.pow(self.theta, powers) #1xd/2
        print("thetas: ", thetas.shape)
        print(token_positions.unsqueeze(dim=-1).shape)
        angles = token_positions.unsqueeze(dim=-1) * thetas[None, :] #[..., seq_len, d/2]
        print("angles: ", angles.shape)
        cos_angles = torch.cos(angles) #...,seq_len,d/2
        # print("cos_angles: ", cos_angles.shape)
        sin_angles = torch.sin(angles) #...,seq_len,d/2
        # print("sin_angles: ", sin_angles.shape)

        rotations = torch.stack([
            torch.stack([cos_angles, sin_angles], dim=-1), #sxdx2
            torch.stack([-sin_angles, cos_angles], dim=-1) #sxdx2
        ], dim=-2) #..., seq_len, d, 2, 2
        print("rotations: ", rotations.shape)
        # print(rotations)


        num_blocks = rotations.shape[-3]  # 4

        block_mask = torch.eye(
            num_blocks,
            device=rotations.device,
            dtype=rotations.dtype,
        )

        out = torch.einsum(
            "...kij,kl->...kilj",
            rotations,
            block_mask,
        )

        out = out.reshape(*rotations.shape[:-3], num_blocks * 2, num_blocks * 2)
        # diag = torch.block_diag(*rotations)
        # print("diag: ", diag.shape)
        print("out: ", out.shape)

        # once i have the rotations, then what?
        # x = 2,6,8. rotations = 2,6,8,8
        # 2,6,1,8 @ 2,6,8,8 -> 2,6,1,8 -> squeeze -> 2,6,8
        roped = x.unsqueeze(dim=-2) @ out
        print("roped @: ", roped.shape)

        roped = roped.squeeze(dim=-2)
        print("roped: ", roped.shape)

        return roped

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        #x is [..., seq_len, token_dim]
        #token_positions is [..., seq_len] - it is the position of the sequence
        #we cannot assume that the sequence positions are 0.. seq_length-1, they are given (i.e. m is given for each sequence)
        #output is the same shape as input, i.e. x.shape

        seq_length, token_dim = x.shape[-2:]
        assert token_dim == self.d_k
        assert seq_length == token_positions.shape[-1]
        assert x.device == self.cos_angles.device

        token_positions = token_positions.to(device=x.device, dtype=torch.long)

        # print("x: ", x.shape)
        # print("thetas: ", thetas.shape)
        cos_angles = self.cos_angles[token_positions].to(x.dtype) #...,seq_len,d/2
        sin_angles = self.sin_angles[token_positions].to(x.dtype) #...,seq_len,d/2

        x = x.reshape(*x.shape[:-1], self.d_k // 2, 2)
        x_top = x[..., 0] #..., d_2. last dimension gets removed when indexing with 0 or 1
        x_bottom = x[..., 1] #..., d_2

        #x_top and x_bottom are pairs
        roped = torch.stack([
            x_top * cos_angles - x_bottom * sin_angles,
            x_top * sin_angles + x_bottom * cos_angles
        ], dim=-1) #..., d_2, 2

        roped = roped.reshape(*x.shape[:-2], self.d_k)

        return roped


class TokenEmbedding(nn.Module):
    def __init__(self, num_embeddings: int, embeddings_dim: int, device=None, dtype=None):
        super().__init__()
        self.mapping = nn.Parameter(torch.empty(num_embeddings, embeddings_dim, device=device, dtype=dtype))
        init.trunc_normal_(self.mapping, mean=0.0, std=1.0, a=-3.0, b=3.0)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # token_ids.shape = [..., d] where we have d integers
        return self.mapping[token_ids]


class Transformer(nn.Module):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()

        self.vocab_size = kwargs.pop("vocab_size")
        self.token_dim = kwargs["token_dim"]
        self.endecoder_layers = kwargs.pop("endecoder_layers")
        self.seq_length = kwargs["seq_length"]
        self.device = kwargs["device"]

        self.tokenEmbeddings = TokenEmbedding(
            num_embeddings=self.vocab_size,
            embeddings_dim=self.token_dim,
            device=self.device
        )
        self.ln = LayerNorm(dim=self.token_dim)

        self.sinusoidalPositionalEmbeddings = self._get_positional_embedding()
        self.rope = RotaryPositionalEmbedding(
            theta = 10000.0,
            d_k = self.token_dim,
            max_seq_length = self.seq_length,
            device=self.device
        )

        self.layers = nn.ModuleList()
        for _ in range(self.endecoder_layers):
            self.layers.append(EnDecoder(**kwargs))

    def _get_positional_embedding(self) -> torch.Tensor:
        x_axis = torch.arange(self.seq_length, dtype=torch.int)
        y_axis_even = torch.arange(0, self.token_dim, 2, dtype=torch.int)
        y_axis_odd = torch.arange(1, self.token_dim, 2, dtype=torch.int)

        float_type = torch.float16

        seq_positions = torch.arange(self.seq_length)[:, None] #4, 1
        powers = 10000 ** (torch.arange(0, self.token_dim, 2) / self.token_dim)
        even_powers = torch.sin(powers)
        odd_powers = torch.cos(powers)

        positional_emb = torch.empty(self.seq_length, self.token_dim, dtype=float_type)
        positional_emb[x_axis[:, None], y_axis_even[None, :]] = (seq_positions / even_powers).to(float_type)
        positional_emb[x_axis[:, None], y_axis_odd[None, :]] = (seq_positions / odd_powers).to(float_type)
        return positional_emb

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # batch_size, seq_len = x.shape

        #ignore the pad tokens.
        #TODO create the input mask
        #padding token also gets learnt, but is useless.
        y = self.tokenEmbeddings(x) #b, seq, token_dim
        # y = y + self.sinusoidalPositionalEmbeddings #seq, token_dim

        for i in range(self.endecoder_layers):
            y = self.layers[i](y)

        #need one more layernorm at the end. Each of the endecoder layers does its own pre-ln.
        y = self.ln(y)
        #now, need to convert the output, back into tokens
        output_token_logits = y @ self.tokenEmbeddings.mapping.T #b, seq, vocab_size
        output_token_probs = Utils.stable_softmax(output_token_logits)

        return output_token_logits, output_token_probs
        

        

# torch.manual_seed(157)
# params = {
#     "vocab_size": 6,
#     "token_dim": 4,
#     "endecoder_layers": 2,
#     "seq_length": 2,
#     "n_heads": 1,
#     "mlp_hidden_layers": 2,
#     "mlp_hidden_layer_dim": 4,
#     "device": "cpu"
# }
# t = Transformer(**params)
# batch_size, seq_length, token_dim = 2, params["seq_length"], params["token_dim"]
# x = torch.randint(low=0, high=params["vocab_size"], size=(batch_size, seq_length))
# print(t(x))

# input_x = torch.rand([batch_size, seq_length, token_dim])
# print("x: ", input_x.shape)
# rope = RotaryPositionalEmbedding(theta = 10.0, d_k=8, max_seq_length=6)
# token_positions = torch.randint_like(torch.empty([batch_size, seq_length]), low=0, high=10)
# print("token_positions: ", token_positions.shape)

# rope(input_x, token_positions)