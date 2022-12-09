import torch
from torch import nn
from checkpoints import CHECKPOINTS
class MarineDebrisDetector(nn.Module):

    def __init__(self):
        self.model = ""
        torch.hub.load("marccoru/marinedebrisdetector", "unetpp")
        pass

    def forward(self):
        pass
def main():
    detector = MarineDebrisDetector()

if __name__ == '__main__':
    main()
