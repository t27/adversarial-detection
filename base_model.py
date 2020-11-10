import torch
import torch.nn as nn


class SkipBlock2(nn.Module):
    """Skip block from resnet paper(Figure2) with additional 1x1 conv on input(to "select" and weight good channels)
    Args:
        nn ([type]): [description]
    """

    def __init__(
        self, in_channels, out_channels,
    ):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(out_channels),
        )
        self.input_conv = nn.Conv2d(
            in_channels=out_channels, out_channels=out_channels, kernel_size=1, stride=1
        )
        self.final_activation = nn.Sequential(nn.ReLU())

    def forward(self, x):
        x = self.block1(x)
        # since our kernels are 3x3, stride1 and pad=1, the size of the feature map should be the same
        x = self.block2(x) + self.input_conv(x)
        x = self.final_activation(x)
        return x


class BaseModel(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # fmt: off
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(16),nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            SkipBlock2(16,16),
            SkipBlock2(16,16),
            SkipBlock2(16,16),
            SkipBlock2(16,16),
            SkipBlock2(16,16),
            # nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(128),nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),nn.BatchNorm2d(32),nn.ReLU(),
            SkipBlock2(32,32),
            SkipBlock2(32,32),
            SkipBlock2(32,32),
            SkipBlock2(32,32),
            SkipBlock2(32,32),
            # nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(128),nn.ReLU(),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),nn.BatchNorm2d(64),nn.ReLU(),
            SkipBlock2(64,64),
            SkipBlock2(64,64),
            SkipBlock2(64,64),
            SkipBlock2(64,64),
            SkipBlock2(64,64),
            # nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(128),nn.ReLU(),
        )

        self.gap = nn.AvgPool2d(kernel_size=8)

        self.linear = nn.Linear(64,num_classes)

        # self.convlayers3 = nn.Sequential(
        #     SkipBlock2(128,128),
        #     SkipBlock2(128,128),
        #     nn.MaxPool2d(2,2),
        #     nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),nn.BatchNorm2d(256),nn.ReLU(),
        #     SkipBlock2(256,256),
        #     SkipBlock2(256,256),
        #     nn.Dropout2d(0.2),
        #     nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1),nn.BatchNorm2d(256),nn.ReLU(),
        #     nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1),nn.BatchNorm2d(256),nn.ReLU(),
        #     nn.MaxPool2d(2,2)
        # )
        # fmt: on

    def forward(self, x, layer_outputs=False):
        c1 = self.conv1(x)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        c4 = self.conv4(c3)
        gap = self.gap(c4)
        gap = torch.flatten(gap, start_dim=1)
        out = self.linear(gap)
        if layer_outputs:
            return out, c1, c2, c3, c4
        else:
            return out


if __name__ == "__main__":
    model = BaseModel()
    im = torch.ones((1, 3, 32, 32))
    model(im)
    # # or
    # outval, l1,l2 = model(img,True)
    # AD1 = AD(1,)

    # out = AD1(l1)
    # loss = criterion(out,target)
