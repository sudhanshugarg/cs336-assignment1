import torch
import torch.nn as nn
import torch.nn.functional as F
from src.tokenizer_v1 import Tokenizer_V1


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


class CrossEntropyFromProbabilities(nn.Module):
    eps = 1e-8
    def __init__(self, device=None):
        super().__init__()
        self.device = device

    def forward(self, probs: torch.Tensor, labels: torch.Tensor):
        #each of the 3 indexes are broadcast into y.shape
        #mask for y also has to be applied.
        #we only take losses from y, for non-padded positions.

        #y.shape = b, seq_length
        # print("labels.shape = ", labels.shape)
        # print("probs.shape = ", probs.shape)
        label_mask = ((labels != Tokenizer_V1.padding_token_int) * 1).to(self.device)
        # print(label_mask)
        losses = probs[torch.arange(probs.shape[0])[:, None], torch.arange(probs.shape[1])[None, :], labels].to(self.device)
        # print("losses.shape = ", losses.shape) #6, 32
        # print(probs[:3, :4])

        # print(losses[:3, :4])
        # print("label_mask.numel() = ", label_mask.numel())
        l2 = losses.reshape(-1)
        # print("l2.numel() = ", l2.numel())
        # print("l2[:10] = ", l2[:10])
        zeros = (l2 == 0).sum()

        loss_sample = losses[0]
        loglosses = torch.log(loss_sample)
        # print("loss_sample = ", loss_sample)
        # print("loglosses = ", loglosses)

        losses = -torch.log(losses + self.eps) * label_mask

        # print("sum loss = ", losses.sum())
        # print("label_mask sum = ", label_mask.sum())
                        
        loss = losses.sum() / (label_mask.sum() + self.eps)

        return loss, zeros



class CrossEntropyFromLogits(nn.Module):
    def __init__(self, device=None):
        super().__init__()
        self.device = device

    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        B, T, V = logits.shape

        loss = F.cross_entropy(
            logits.reshape(B * T, V),
            labels.reshape(B * T),
            ignore_index=Tokenizer_V1.padding_token_int,
        )
        return loss

