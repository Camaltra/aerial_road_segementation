import torch
import torch.nn as nn
import torchvision.transforms.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_c: int, out_c: int) -> None:
        super().__init__()
        self.first = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.batch_norm_first = nn.BatchNorm2d(out_c)
        self.act_fn_first = nn.ReLU(inplace=True)

        self.second = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.batch_norm_second = nn.BatchNorm2d(out_c)
        self.act_fn_second = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.first(x)
        x = self.batch_norm_first(x)
        x = self.act_fn_first(x)
        x = self.second(x)
        x = self.batch_norm_second(x)
        return self.act_fn_second(x)


class DownSampling(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.max_pool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.max_pool(x)


class UpSampling(nn.Module):
    def __init__(self, in_c: int, out_c: int) -> None:
        super().__init__()
        self.up_sampling = nn.ConvTranspose2d(in_channels=in_c, out_channels=out_c, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up_sampling(x)


class CropAndConcat(nn.Module):
    def forward(self, x: torch.Tensor, contracting_x: torch.Tensor) -> torch.Tensor:
        contracting_x = F.center_crop(contracting_x, [x.shape[2], x.shape[3]])
        return torch.concat((x, contracting_x), dim=1)


class UNet(nn.Module):
    def __init__(self, in_channel: int, out_channel: int) -> None:
        super().__init__()
        self.down_convs = nn.ModuleList([DoubleConv(in_c, out_c) for in_c, out_c in [(in_channel, 64), (64, 128), (128, 256), (256, 512)]])
        self.down_samples = nn.ModuleList([DownSampling() for _ in range(4)])

        self.middle_conv = nn.Conv2d(512, 1024, kernel_size=3, padding=1)

        self.up_samples = nn.ModuleList([UpSampling(in_c, out_c) for in_c, out_c in [(1024, 512), (512, 256), (256, 128), (128, 64)]])
        self.up_convs = nn.ModuleList([DoubleConv(in_c, out_c) for in_c, out_c in [(1024, 512), (512, 256), (256, 128), (128, 64)]])

        self.final_conv = nn.Conv2d(64, out_channel, kernel_size=1)

        self.crop_concat = CropAndConcat()

    def forward(self, x) -> torch.Tensor:
        skip_connection = []

        for down_conv, down_sample in zip(self.down_convs, self.down_samples):
            x = down_conv(x)
            skip_connection.append(x)
            x = down_sample(x)

        x = self.middle_conv(x)

        for up_conv, up_sample in zip(self.up_convs, self.up_samples):
            x = up_sample(x)
            y = skip_connection.pop()
            x = self.crop_concat(x, y)
            x = up_conv(x)

        return self.final_conv(x)