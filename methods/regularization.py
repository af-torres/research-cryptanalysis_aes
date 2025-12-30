import torch.nn as nn

class LockedDropout(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0:
            return x
        mask = x.new_empty(1, x.size(1), x.size(2)).bernoulli_(1 - self.p)
        mask = mask / (1 - self.p)
        return x * mask