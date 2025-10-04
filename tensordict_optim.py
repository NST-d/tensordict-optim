import torch
import tensordict
from typing import Optional, overload


@torch.no_grad()
def zero_grad(p):
    if p.grad is not None:
        p.grad.zero_()


class Optimizer:

    def step(self, params: tensordict.TensorDict, lr: float):
        raise NotImplementedError

class SGD(Optimizer):
    def __init__(self, weight_decay: Optional[float] = None):
        self.weight_decay = weight_decay

    @torch.no_grad()
    def step(self, params, lr):
        params.add_(params.grad, alpha=-lr)
        if self.weight_decay is not None:
            params.add_(params, alpha=-lr * self.weight_decay)

class SGDmoment(Optimizer):
    def __init__(self, param_dict, tau=0.9, weight_decay=None):
        self.moment = torch.zeros_like(param_dict)
        self.tau = tau
        self.weight_decay = weight_decay

    @torch.no_grad()
    def step(self, params, lr):
        self.moment.add_(params.grad, alpha=1-self.tau)
        params.add_(self.moment, alpha=-lr)
        if self.weight_decay is not None:
            params.add_(params, alpha=-lr * self.weight_decay)

class Nesterov(Optimizer):
    def __init__(self, param_dict, rho=0.9, weight_decay=None):
        self.moment = torch.zeros_like(param_dict)
        self.rho = rho
        self.weight_decay = weight_decay

    @torch.no_grad()
    def step(self, params, lr):
        new_moment = self.moment.mul(self.rho).add(params.grad, alpha=-lr)
        params.add_(self.moment.mul(-self.rho).add_(new_moment, alpha=1 + self.rho))
        self.moment = new_moment
        if self.weight_decay is not None:
            params.add_(params, alpha=-lr * self.weight_decay)

class AdaGrad(Optimizer):
    def __init__(self, params, weight_decay=None, eps=1e-8):
        self.moment = torch.zeros_like(params)
        self.eps = eps
        self.weight_decay = weight_decay


    @torch.no_grad()
    def step(self, params, lr, frozen_moment=False):
        self.moment.add_(params.grad.pow(2))
        params.addcdiv_(params.grad.div, self.moment.sqrt() + self.eps, value=-lr)
        if self.weight_decay is not None:
            params.add_(params, alpha=-lr * self.weight_decay)

class RMSProp(Optimizer):
    def __init__(self, params, alpha=0.99, weight_decay=None, eps=1e-8):
        self.moment = torch.zeros_like(params)
        self.eps = eps
        self.weight_decay = weight_decay
        self.alpha = alpha

    @torch.no_grad()
    def step(self, params, lr):
        self.moment.lerp_(params.grad.pow(2), 1 - self.alpha)
        params.addcdiv_(params.grad, self.moment.sqrt() + self.eps, value=-lr)

        if self.weight_decay is not None:
            params.add_(params, alpha=-lr * self.weight_decay)

class Adam(Optimizer):
    def __init__(self, param_dict, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2):
        self.moment = torch.zeros_like(param_dict)
        self.second_moment = torch.zeros_like(param_dict)

        self.beta1 = betas[0]
        self.beta2 = betas[1]

        self.eps = eps
        self.t = 0

        self.weight_decay = weight_decay

    @torch.no_grad()
    def step(self, params, lr):
        self.moment.lerp_(params.grad, 1 - self.beta1)
        self.second_moment.lerp_(params.grad.pow(2), 1 - self.beta2)
        self.t += 1
        scaled_moment = self.moment.div(1 - self.beta1 ** self.t)
        scaled_second_moment = self.second_moment.div(1 - self.beta2 ** self.t)

        params.add_(scaled_moment.div(scaled_second_moment.sqrt() + self.eps), alpha=-lr)

        # weight decay
        if self.weight_decay is not None:
            params.add_(params, alpha=-lr * self.weight_decay)



