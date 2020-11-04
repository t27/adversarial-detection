import torch
import torch.nn as nn


# TODO: just a placeholder
class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        self.conv1 = None
        self.conv2 = None
        self.final = None

    def forward(self, x, layer_outputs=False):
        l1 = self.conv1(x)
        l2 = self.conv2(l1)
        out = self.final(l2)
        if layer_outputs:
            return out, l1, l2
        else:
            return out


if __name__ == "__main__":
    model = BaseModel()
    model.eval()
    img = []
    outval = model(img)
    # # or
    # outval, l1,l2 = model(img,True)
    # AD1 = AD(1,)

    # out = AD1(l1)
    # loss = criterion(out,target)
