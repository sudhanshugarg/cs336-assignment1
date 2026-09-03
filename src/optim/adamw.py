import torch.nn as nn
from torch.optim import Optimizer
from collections.abc import Callable
from typing import Optional
import torch


class AdamWCls(Optimizer):
    def __init__(self, params, lr, betas, weight_decay, eps):
        self.eps = eps
        self.weight_decay = weight_decay
        defaults = {
            "lr": lr,
            "beta1": betas[0],
            "beta2": betas[1],
            "eps": eps,
            "weight_decay": weight_decay      
        }
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if not closure else closure()

        """
        momentum
        v_t = beta1 * v_(t-1) + (1-beta1) * dl/dw
        w -= lr * v_t

        rmsprop
        s_t = beta2 * s_(t-1) + (1-beta2) * dl/dw**2
        adam
        w -= (lr / (sqrt(s_t) + eps)) * v_t

        adamw
        w = w - ( lr * 
                       (    
                            (v_t / (sqrt(s_t) + eps))
                       ) + 
                       (weight_decay * w)
                )
        """

        groups = self.param_groups
        for group in groups:
            lr = group["lr"]
            beta1 = group["beta1"]
            beta2 = group["beta2"]
            params = group["params"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            loop = 0
            for p in params:
                loop += 1
                if p.grad is None:
                    continue

                state = self.state[p]
                t = state.get("t", 0)
                v_t = state.get("v_t", 0.0)
                s_t = state.get("s_t", 0.0)
                grad = p.grad.data

                # print("loop = ", loop, ", data shape = ", p.data.shape, ", grad shape = ", p.grad.data.shape)

                # momentum
                v_t = (beta1 * v_t) + ((1 - beta1) * grad)

                # rmsprop
                s_t = (beta2 * s_t) + ((1 - beta2) * (torch.pow(grad, 2)))

                # print("vt, st shape = ", v_t.shape, s_t.shape)

                # unbias them
                v_t_unbiased = v_t / (1.0 - (beta1 ** (t+1)))
                s_t_unbiased = s_t / (1.0 - (beta2 ** (t+1)))

                # print("(unbiased) vt, st shape = ", v_t_unbiased.shape, s_t_unbiased.shape)
                # print("(unbiased) vt = ", v_t_unbiased)

                # final update
                p.data -= lr * (
                    (v_t_unbiased / (torch.sqrt(s_t_unbiased) + eps)) +
                    weight_decay * p.data
                )

                state["t"] = t + 1
                state["v_t"] = v_t
                state["s_t"] = s_t

        return loss


if __name__ == "__main__":
    weights = nn.Parameter(torch.randn(10, 10) * 5)
    optimizer = AdamWCls(lr=1.0, params=[weights], betas=(0.9, 0.95), weight_decay=1e-3, eps=1e-8)
    for t in range(10):
        optimizer.zero_grad()
        loss = (weights ** 2).mean()
        print(loss.cpu().item())
        loss.backward()
        optimizer.step()