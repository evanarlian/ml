import torch
from torch import nn
from torchinfo import summary


class Convolutional(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int, stride: int):
        super().__init__()
        # https://github.com/pjreddie/darknet/blob/master/cfg/yolov1.cfg
        # I think batch_normalize=1 and pad=1 are booleans, not integers
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.kernel = kernel
        self.stride = stride
        self.padding = kernel // 2
        self.conv = nn.Conv2d(in_ch, out_ch, kernel, stride, self.padding)
        self.bn = nn.BatchNorm2d(num_features=out_ch)
        self.leaky = nn.LeakyReLU(negative_slope=0.1)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.leaky(x)
        return x


class YoloBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # source: see yolo paper on figure 3 (only the convs)
        self.group1 = nn.Sequential(
            Convolutional(3, 64, kernel=7, stride=2),
            nn.MaxPool2d(2, stride=2),
        )
        self.group2 = nn.Sequential(
            Convolutional(64, 192, kernel=3, stride=1),
            nn.MaxPool2d(2, stride=2),
        )
        self.group3 = nn.Sequential(
            Convolutional(192, 128, kernel=1, stride=1),
            Convolutional(128, 256, kernel=3, stride=1),
            Convolutional(256, 256, kernel=1, stride=1),
            Convolutional(256, 512, kernel=2, stride=1),
            nn.MaxPool2d(2, stride=2),
        )
        self.group4 = nn.Sequential(
            # 2 convs copied 4 times
            Convolutional(512, 256, kernel=1, stride=1),
            Convolutional(256, 512, kernel=3, stride=1),
            Convolutional(512, 256, kernel=1, stride=1),
            Convolutional(256, 512, kernel=3, stride=1),
            Convolutional(512, 256, kernel=1, stride=1),
            Convolutional(256, 512, kernel=3, stride=1),
            Convolutional(512, 256, kernel=1, stride=1),
            Convolutional(256, 512, kernel=3, stride=1),
            # 2 remaining convs
            Convolutional(512, 512, kernel=1, stride=1),
            Convolutional(512, 1024, kernel=3, stride=1),
            nn.MaxPool2d(2, stride=2),
        )
        self.group5 = nn.Sequential(
            # 2 convs copied 2 times
            Convolutional(1024, 512, kernel=1, stride=1),
            Convolutional(512, 1024, kernel=3, stride=1),
            Convolutional(1024, 512, kernel=1, stride=1),
            Convolutional(512, 1024, kernel=3, stride=1),
            # 2 remaining convs
            Convolutional(1024, 1024, kernel=3, stride=1),
            Convolutional(1024, 1024, kernel=3, stride=2),
        )
        self.group6 = nn.Sequential(
            Convolutional(1024, 1024, kernel=3, stride=1),
            Convolutional(1024, 1024, kernel=3, stride=1),
        )

    def forward(self, x):
        x = self.group1(x)
        x = self.group2(x)
        x = self.group3(x)
        x = self.group4(x)
        x = self.group5(x)
        x = self.group6(x)
        return x

    def debug_forward(self, x):
        # fmt: off
        x = self.group1(x); print("after group1", x.size())
        x = self.group2(x); print("after group2", x.size())
        x = self.group3(x); print("after group3", x.size())
        x = self.group4(x); print("after group4", x.size())
        x = self.group5(x); print("after group5", x.size())
        x = self.group6(x); print("after group6", x.size())
        # fmt: off
        return x


class YoloPretraining(nn.Module):
    def __init__(self, n_classes: int):
        super().__init__()
        self.n_classes = n_classes
        self.backbone = YoloBackbone()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(1024 * 4 * 4, n_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

    def debug_forward(self, x):
        # fmt: off
        x = self.backbone(x); print("after backbone", x.size())
        x = self.flatten(x); print("after flatten", x.size())
        x = self.fc(x); print("after fc", x.size())
        # fmt: off
        return x


class YoloDetection(nn.Module):
    def __init__(self, B: int, C: int):
        super().__init__()
        # source: see yolo paper on figure 3 (all, even with fc)
        self.B = B  # bboxes per grid
        self.C = C  # classes per grid
        self.n_items = B * 5 + C
        self.backbone = YoloBackbone()
        self.detector = nn.Sequential(
            nn.Linear(1024, 4096),
            nn.LeakyReLU(),
            nn.Linear(4096, self.n_items),
        )

    def forward(self, x):
        x = self.backbone(x)
        x = x.permute(0, 2, 3, 1)
        x = self.detector(x)
        return x

    def debug_forward(self, x):
        # fmt: off
        x = self.backbone(x); print("after backbone", x.size())
        x = x.permute(0, 2, 3, 1); print("after permute", x.size())
        x = self.detector(x); print("after conv2fc", x.size())
        # fmt: off
        return x


def main():
    # fake data, halved during pretraining
    x_detection = torch.randn(2, 3, 448, 448)
    x_pretrain = torch.randn(2, 3, 224, 224)

    # backbone
    yolo_backbone = YoloBackbone()
    summary(yolo_backbone, input_data=x_detection, depth=1)
    yolo_backbone.debug_forward(x_detection)
    print()

    # with pretraining head (can be imagenet)
    yolo_pretraining = YoloPretraining(n_classes=1000)
    summary(yolo_pretraining, input_data=x_pretrain, depth=2)
    yolo_pretraining.debug_forward(x_pretrain)
    print()

    # with detection head
    yolo_detection = YoloDetection(B=2, C=20)
    summary(yolo_detection, input_data=x_detection, depth=2)
    yolo_detection.debug_forward(x_detection)
    print()


if __name__ == "__main__":
    main()
