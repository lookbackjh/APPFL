import torch
from torch.nn.modules.loss import _Loss


class BCEWithLogitsLoss(_Loss):
    def __init__(self):
        super().__init__()
        self.bce_logit_loss = torch.nn.BCEWithLogitsLoss()

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        #return self.bce(input, target)
        return self.bce_logit_loss(input, target)   