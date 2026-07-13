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
    _dtype_map = {
        "float16": torch.float16,
        "float32": torch.float32,
        "float64": torch.float64,
        "bfloat16": torch.bfloat16,
        "int8": torch.int8,
        "int16": torch.int16,
        "int32": torch.int32,
        "int64": torch.int64,
        "bool": torch.bool,
    }

    @staticmethod
    def stable_softmax(x: torch.Tensor, dimension: int=-1) -> torch.Tensor:
        max_logit = torch.max(x, dim=dimension, keepdim=True).values
        x = torch.exp(x - max_logit) #b, n_heads, seq, seq
        sum = torch.sum(x, dim=dimension, keepdim=True)
        return x / sum

    @staticmethod
    def get_dtype(type: str | None = None):
        if type is not None and type in Utils._dtype_map:
            return Utils._dtype_map[type]
        return None


class EnDecoder(nn.Module):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        using_dtype = Utils.get_dtype(kwargs["dtype"])
        self.attention = CausalSelfAttention(**kwargs)
        self.mlp = FFN(d_ff=kwargs["d_ff"], d_model=kwargs["token_dim"], dtype=using_dtype)
        self.device = kwargs["device"]
        self.norm1 = RMSNorm(d_model=kwargs["token_dim"], device=self.device, dtype=using_dtype)
        self.norm2 = RMSNorm(d_model=kwargs["token_dim"], device=self.device, dtype=using_dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #PreLN norm residual
        # x1 = x.clone()
        # x1 = self.ln(x1)
        # x1 = self.attention(x1, True)
        # x1 = x + x1
        # print("in endecoder")
        seq = x.shape[-2]
        token_positions = torch.arange(seq, device=self.device)
        x = x + self.attention(self.norm1(x), token_positions=token_positions, use_upper_triangular=True)

        # x2 = x1.clone()
        # x2 = self.ln(x2)
        # x2 = self.mlp(x2)
        x = x + self.mlp(self.norm2(x))
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
    def __init__(self, d_ff: int, d_model: int, dtype=None) -> None:
        super().__init__()
        self.d_ff = d_ff
        self.d_model = d_model
        self.w1 = Linear2(in_features=d_model, out_features=d_ff, dtype=dtype)
        self.w2 = Linear2(in_features=d_ff, out_features=d_model, dtype=dtype)
        self.w3 = Linear2(in_features=d_model, out_features=d_ff, dtype=dtype)
        self.silu = SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[-1] == self.d_model
        # print("in ffn")
        w1_x = self.w1(x) # ..., d_ff
        w1_x = self.silu(w1_x)
        w3_x = self.w3(x) # ..., d_ff
        w2_x = self.w2(w1_x * w3_x)
        return w2_x

class Linear2(nn.Module):
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None) -> None:
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

@DeprecationWarning
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
        using_dtype = Utils.get_dtype(kwargs["dtype"])

        self.Q = Linear2(self.token_dim, self.token_dim, dtype=using_dtype)
        self.K = Linear2(self.token_dim, self.token_dim, dtype=using_dtype)
        self.V = Linear2(self.token_dim, self.token_dim, dtype=using_dtype)
        self.up_proj = Linear2(self.token_dim, self.token_dim, dtype=using_dtype)

        max_seq_length = kwargs["max_seq_length"]
        device = kwargs["device"]
        theta = float(kwargs.get("theta", 10000.0))
        self.rope = RotaryPositionalEmbedding(
            theta=theta,
            d_k=self.head_dim,
            max_seq_length=max_seq_length,
            device=device,
            dtype=using_dtype)

    def _lower_triangular(self, n: int) -> torch.Tensor:
        rows = torch.arange(n).view(n, 1)
        cols = torch.arange(n).view(-1, n)
        # we want 1s to represent the values to keep, and 0s to represent
        # the values to set to -inf. hence return negation.
        return ~(rows < cols)

    @staticmethod
    def scaled_dot_product_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        #q.shape = ..., b, n_heads, seq, head_dim
        attention = q @ k.transpose(-2, -1) / math.sqrt(float(q.shape[-1])) #b, n_heads, seq, seq
        # print("within scaled dot: ", attention.shape)
        if mask is not None:
            attention = attention.masked_fill(~mask, float("-inf"))

        attention_softmax = Utils.stable_softmax(attention, dimension=-1) #b, n_heads, seq, seq
        result = torch.matmul(attention_softmax, v) #b, n_heads, seq, head_dim

        return result

    def forward(self,
                x: torch.Tensor,
                token_positions: torch.Tensor | None = None,
                mask: torch.Tensor | None = None,
                use_upper_triangular: bool = False) -> torch.Tensor:
        seq = x.shape[-2]
        if token_positions is not None:
            assert seq == token_positions.shape[-1]
        if mask is not None:
            assert seq == mask.shape[-1]

        # print("in selfattn")
        q = self.Q(x) #b, seq, token
        k = self.K(x) #b, seq, token
        v = self.V(x) #b, seq, token
        # print("q,k,v.shape: ", q.shape, k.shape, v.shape)

        q = q.reshape(*x.shape[:-1], self.n_heads, self.head_dim).transpose(-2, -3) #b, n_heads, seq, head_dim
        k = k.reshape(*x.shape[:-1], self.n_heads, self.head_dim).transpose(-2, -3)

        if token_positions is not None:
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)

        v = v.reshape(*x.shape[:-1], self.n_heads, self.head_dim).transpose(-2, -3)
        # print("q,k,v.shape (after reshape): ", q.shape, k.shape, v.shape)

        if mask is None and use_upper_triangular:
            mask = self._lower_triangular(seq)
        attention = CausalSelfAttention.scaled_dot_product_attention(q, k, v, mask)
        # print("scaled dot product attention shape: ", attention.shape)
        attention = attention.transpose(-2, -3)
        # print("scaled dot product attention reshape: ", attention.shape)
        result = self.up_proj(attention.reshape(x.shape))
        # print("result shape: ", result.shape)
        return result

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
        # print("in rmsnorm")
        in_dtype = x.dtype
        x = x.to(torch.float32) #just because we are squaring, so keep additional precision.
        #what is rms norm
        denominator = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return ((x / denominator) * self.gamma).to(in_dtype)

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_length: int, device=None, dtype=None):
        super().__init__()
        assert d_k % 2 == 0
        self.d_k = d_k
        self.d_k_2 = d_k // 2
        self.theta = theta
        self.max_seq_length = max_seq_length
        self.device = device

        # for pre-computing the angles for all positions
        powers = -2.0 * torch.arange(self.d_k // 2, dtype=dtype, device=device) / self.d_k
        thetas = torch.pow(theta, powers) #1xd/2
        # print(token_positions.unsqueeze(dim=-1).shape)
        angles = torch.arange(max_seq_length, device=device, dtype=dtype).unsqueeze(dim=-1) * thetas[None, :] #[..., seq_len, d/2]
        # print("angles: ", angles.shape)
        cos = torch.cos(angles) #...,seq_len,d/2
        # print("cos_angles: ", cos_angles.shape)
        sin = torch.sin(angles) #...,seq_len,d/2

        # self.register_buffer("cos_angles", cos)
        # self.register_buffer("sin_angles", sin)
        self.cos_angles = cos
        self.sin_angles = sin

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
        assert token_dim == self.d_k, f"token_dim = {token_dim}, self.d_k = {self.d_k}"
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

class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, seq_length: int, token_dim: int):
        self.seq_length = seq_length
        self.token_dim = token_dim
        self.positional_embedding = self._get_positional_embedding()

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


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[-1] == self.token_dim
        assert x.shape[-2] == self.seq_length

        return x + self.positional_embedding


class TokenEmbedding(nn.Module):
    def __init__(self, num_embeddings: int, embeddings_dim: int, device=None, dtype=None):
        super().__init__()
        self.mapping = nn.Parameter(torch.empty(num_embeddings, embeddings_dim, device=device, dtype=dtype))
        init.trunc_normal_(self.mapping, mean=0.0, std=1.0, a=-3.0, b=3.0)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # token_ids.shape = [..., d] where we have d integers
        return self.mapping[token_ids]

"""
FLOPS & Params
b = batch, s = seq, t = token_dim, h = head_dim, n = num_heads
self.tokenEmbeddings
  - params: tv
  - flops: 0
EnDecoder:
  - norm1
    - params: t
    - flops: 6 * bst, bsn*h = x.numel()
  - attention
    - params: 4 * t**2
    - flops: 4 * bs*t^2 + 2 * bn*s*h*s + b*n*s^2 + 3bns^2 + 2 * bnh * s^2 + 6bnsh
  - norm2
    - params: t
    - flops: 6 * bst, bsn*h = x.numel()
  - mlp
    - params: 3 * t * d_ff
    - flops: 2*bst*d_ff + 2*bst*d_ff + 2*bst*d_ff + b*s*d_ff + 2*bs d_ff * t + 2bs * d_ff

final_norm
    - params: t
    - flops: 6 * bst

lm_head
    - params: t * v
    - flops: 2 * bstv + 3bsv


Total:
 params: layers * (2t + 4t^2 + 3t*d_ff) + t + 2tv
 flops: layers * (18bst + 4bs t^2 + 4 bt s^2 + 4bn s^2 + 8 bst d_ff + 3 bs d_ff) + 6bst + 2bstv + 3bsv
"""
class Transformer(nn.Module):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()

        self.vocab_size = kwargs.pop("vocab_size")
        self.token_dim = kwargs["token_dim"]
        self.endecoder_layers = kwargs.pop("endecoder_layers")
        self.device = kwargs["device"]
        using_dtype = Utils.get_dtype(kwargs["dtype"])

        self.tokenEmbeddings = TokenEmbedding(
            num_embeddings=self.vocab_size,
            embeddings_dim=self.token_dim,
            device=self.device,
            dtype=using_dtype
        )
        self.norm = RMSNorm(d_model=self.token_dim, device=self.device, dtype=using_dtype)
        self.lm_head = Linear2(
            in_features=self.token_dim,
            out_features=self.vocab_size,
            device=self.device,
            dtype=using_dtype
        )

        self.layers = nn.ModuleList()
        for _ in range(self.endecoder_layers):
            self.layers.append(EnDecoder(**kwargs))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # batch_size, seq_len = x.shape

        #ignore the pad tokens.
        #TODO create the input mask
        #padding token also gets learnt, but is useless.
        y = self.tokenEmbeddings(x) #b, seq, token_dim
        # y = y + self.sinusoidalPositionalEmbeddings #seq, token_dim

        # print("after token_embedding")
        for i in range(self.endecoder_layers):
            y = self.layers[i](y)
            # print(f"after layer {i}")


        #need one more layernorm at the end. Each of the endecoder layers does its own pre-ln.
        y = self.norm(y)
        # print(f"after norm")
        #now, need to convert the output, back into tokens
        #TODO try weight tying
        # output_token_logits = y @ self.tokenEmbeddings.mapping.T #b, seq, vocab_size
        output_token_logits = self.lm_head(y)
        # print(f"after lm_head")
        output_token_probs = Utils.stable_softmax(output_token_logits)
        # print(f"after softmax")

        return output_token_logits, output_token_probs

def compute_params_and_flops(
    layers: int,
    b: int,
    s: int,
    t: int,
    v: int,
    d_ff: int,
    n: int
) -> tuple[int, int]:
    """
    Compute the parameter count and FLOP count for the given model dimensions.

    Note:
        h is included in the interface but does not appear in the supplied
        formulas.
    """
    params = layers * (2 * t + 4 * t**2 + 3 * t * d_ff) + t + 2 * t * v

    flops = (
        layers
        * (
            18 * b * s * t
            + 4 * b * s * t**2
            + 4 * b * t * s**2
            + 4 * b * n * s**2
            + 8 * b * s * t * d_ff
            + 3 * b * s * d_ff
        )
        + 6 * b * s * t
        + 2 * b * s * t * v
        + 3 * b * s * v
    )

    return params, flops

params, flops = compute_params_and_flops(
    layers=48,
    b=1,
    s=1024,
    t=1600,
    v=50_257,
    d_ff=4288,
    n=25
)
"""
vocab_size:  50,257
context_length:  1,024
num_layers:  48
d_model:  1,600
num_heads:  25
d_ff:  4,288 (the nearest multiple of 64 to 8/3 × 1, 600)
"""
# print(f"Parameters: {params:,}")
# print(f"FLOPs:      {flops:,}")

torch.manual_seed(157)
# params = {
#     "vocab_size": 3,
#     "token_dim": 64,
#     "endecoder_layers": 2,
#     "max_seq_length": 16,
#     "n_heads": 4,
#     "d_ff": 4,
#     "theta": 10000,
#     "dtype": "float16",
#     "device": "cpu",
# }
# t = Transformer(**params)
# batch_size, seq_length, token_dim = 5, 7, params["token_dim"]
# assert seq_length <= params["max_seq_length"]
# x = torch.randint(low=0, high=params["vocab_size"], size=(batch_size, seq_length))
# logits, probs = t(x)
# print(logits.shape)
# print(logits)

# input_x = torch.rand([batch_size, seq_length, token_dim])
# print("x: ", input_x.shape)
# rope = RotaryPositionalEmbedding(theta = 10.0, d_k=8, max_seq_length=6)
# token_positions = torch.randint_like(torch.empty([batch_size, seq_length]), low=0, high=10)
# print("token_positions: ", token_positions.shape)

# rope(input_x, token_positions)