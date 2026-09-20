import pprint
from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math
import torch
import torch.nn as nn
import numpy as np
import random

ppr = pprint.PrettyPrinter()


class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"] # Get the learning rate.
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p] # Get state associated with p.
                t = state.get("t", 0) # Get iteration number from the state, or initial value.
                grad = p.grad.data # Get the gradient of loss with respect to p.
                p.data -= lr / math.sqrt(t + 1) * grad # Update weight tensor in-place.
                state["t"] = t + 1 # Increment iteration number.
        return loss


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01, device='cpu'):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")

        defaults = {"lr": lr,
                    'betas': betas,
                    'eps': eps,
                    'weight_decay': weight_decay}
        self.device = device

        super().__init__(params, defaults)

        for i, group in zip(range(len(self.param_groups)), self.param_groups):
            # moments = {'moments1': [],
            #            'moments2': []}
            for j, p in zip(range(len(group['params'])), group['params']):
                # moments['moments1'].append(nn.Parameter(torch.zeros(p.shape)).to(torch.device(device)))
                # moments['moments2'].append(nn.Parameter(torch.zeros(p.shape)).to(torch.device(device)))
                self.state[p] = {'moment1': nn.Parameter(torch.zeros(p.shape)).to(torch.device(device)),
                                 'moment2': nn.Parameter(torch.zeros(p.shape)).to(torch.device(device))}

        # ppr.pprint('AFTER INIT:')
        # ppr.pprint(self.state_dict())

    def to(self, device):
        for i, group in zip(range(len(self.param_groups)), self.param_groups):
            for j, p in zip(range(len(group['params'])), group["params"]):
                self.state[p]['moment1'] = self.state[p]['moment1'].to(torch.device(device))
                self.state[p]['moment2'] = self.state[p]['moment2'].to(torch.device(device))

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()

        for i, group in zip(range(len(self.param_groups)), self.param_groups):
            lr = group["lr"]

            for j, p in zip(range(len(group['params'])), group["params"]):
                if p.grad is None:
                    continue

                p_state = self.state[p]

                t = p_state.get("t", 1)
                grad = p.grad.data

                # self.state[i]['moments1'][j] = group['betas'][0] * self.state[i]['moments1'][j] + (1 - group['betas'][0]) * grad
                # self.state[i]['moments2'][j] = group['betas'][1] * self.state[i]['moments2'][j] + (1 - group['betas'][1]) * torch.square(grad)

                self.state[p]['moment1'] = group['betas'][0] * self.state[p]['moment1'] + (
                            1 - group['betas'][0]) * grad
                self.state[p]['moment2'] = group['betas'][1] * self.state[p]['moment2'] + (
                            1 - group['betas'][1]) * torch.square(grad)

                adj_lr = lr * ((math.sqrt(1 - group['betas'][1] ** t)) / (1 - group['betas'][0] ** t))

                # p.data -= adj_lr * (self.state[i]['moments1'][j] / (torch.sqrt(self.state[i]['moments2'][j]) + group['eps']))
                # p.data -= lr * group['weight_decay'] * p.data

                p.data -= adj_lr * (
                            self.state[p]['moment1'] / (torch.sqrt(self.state[p]['moment2']) + group['eps']))
                p.data -= lr * group['weight_decay'] * p.data

                p_state["t"] = t + 1

        return loss


def lr_cosine_schedule(t, lr_max, lr_min, t_warmup, t_cooldown):
    if t < t_warmup:
        return (t / t_warmup) * lr_max
    elif t_warmup <= t <= t_cooldown:
        return lr_min + 0.5 * (1 + math.cos(((t - t_warmup) / (t_cooldown - t_warmup)) * math.pi)) * (lr_max - lr_min)
    else:
        return lr_min


def clip_gradients(params, max_norm, norm=2, eps=1e-6):
    for param in params:
        grad_norm = torch.norm(param.grad, p=norm)
        param.grad = param.grad if grad_norm < max_norm else (max_norm / (grad_norm + eps)) * param.grad


def load_batch(arr, batch_size, context_length, device='cpu'):
    random_idx_batch = torch.randint(low=0, high=len(arr) - context_length, size=(batch_size,))
    return torch.tensor(np.array([arr[i:i+context_length] for i in random_idx_batch])).to(torch.device(device)), torch.tensor(np.array([arr[i+1:i+context_length+1] for i in random_idx_batch])).to(torch.device(device))

    # return examples[random_idx_batch], labels[random_idx_batch]

    # examples = torch.tensor(np.array([arr[i:i + context_length] for i in range(0, len(arr) - context_length, 1)])).to(
    #     torch.device(device))
    # labels = torch.tensor(np.array([arr[i:i + context_length] for i in range(1, len(arr) + 1 - context_length, 1)])).to(
    #     torch.device(device))

    # random_start = random.randint(0, len(arr) - context_length - 1)
    # if random_start <= len(arr) - context_length - batch_size:
    #     return examples[random_start:random_start + batch_size], labels[random_start:random_start + batch_size]
    # else:
    #     boundary = len(arr) - context_length
    #     remainder = (random_start + batch_size) % boundary
    #     # print(random_start, boundary, 0, remainder)
    #     return torch.cat((examples[random_start:boundary], examples[0:remainder]), dim=0).to(torch.device(device)), torch.cat((labels[random_start:boundary], labels[0:remainder]), dim=0).to(torch.device(device))


def save_checkpoint(model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer,
                    iteration: int,
                    out):
    # print('ABOUT TO SAVE:')
    # ppr.pprint(optimizer.state_dict())
    torch.save({'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'training iteration': iteration},
               out)


def load_checkpoint(src,
                    model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer):
    model.load_state_dict(torch.load(src)['model_state'])
    optimizer.load_state_dict(torch.load(src)['optimizer_state'])

    return torch.load(src)['training iteration']


def toy_exp(lr, its):
    weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
    opt = SGD([weights], lr=lr)

    print('-'*50)
    print('lr=', lr, 'its=', its)

    for t in range(its):
        opt.zero_grad()  # Reset the gradients for all learnable parameters.
        loss = (weights ** 2).mean()  # Compute a scalar loss value.
        print(loss.cpu().item())
        loss.backward()  # Run backward pass, which computes gradients.
        opt.step()  # Run optimizer step.

    print('-'*50)


def main():
    lrs = [1, 1e1, 1e2, 1e3]
    print('Starting exps...')
    for lr in lrs:
        toy_exp(lr=lr, its=10)


if __name__ == '__main__':
    main()

