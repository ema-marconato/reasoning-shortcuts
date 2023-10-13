import torch


class ADDMNIST_DPL(torch.nn.Module):
    def __init__(self, loss, nr_classes=19) -> None:
        super().__init__()
        self.base_loss = loss
        self.nr_classes = nr_classes

    def forward(self, out_dict, args): 
        loss, losses = self.base_loss(out_dict, args)
        return loss, losses