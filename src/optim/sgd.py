from torch.optim import Optimizer
from typing import Optional
from collections.abc import Callable
import math
import torch
import torch.nn as nn


class SGD(Optimizer):
    def __init__(self, params, lr) -> None:
        self.params = params
        self.lr = lr
        defaults = {
            "lr": lr
        }
        super().__init__(params, defaults)


    def step(self, closure: Optional[Callable] = None):
        loss = None
        if closure:
            loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            params = group["params"]
            for p in params:

                if p.grad is None:
                    continue

                state = self.state[p]
                t = state.get("t", 0)
                grad = p.grad.data

                p.data = p.data - (lr * grad) / math.sqrt(t + 1)
                state["t"] = t + 1

        return loss

if __name__ == "__main__":
    weights = nn.Parameter(torch.randn(10, 10) * 5)
    optimizer = SGD(lr=1000.0, params=[weights])
    for t in range(10):
        optimizer.zero_grad()
        loss = (weights ** 2).mean()
        print(loss.cpu().item())
        loss.backward()
        optimizer.step()
