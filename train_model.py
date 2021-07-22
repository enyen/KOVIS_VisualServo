import torch
import torch.nn as nn
import torch.nn.functional as F
from train_densenet import DenseNet, DenseNetTranspose, DenseBlock, TransitionDown, TransitionUp


class Encoder(nn.Module):
    def __init__(self, num_input, num_keypoint, growth_rate, block_cfg, drop_rate, kper, inference=False):
        super(Encoder, self).__init__()
        self.inference = inference

        num_outputs = []
        # downstream
        num_inputs = [num_input, num_input + growth_rate * block_cfg[0][0],
                      (num_input + growth_rate * block_cfg[0][0]) // 2]
        num_outputs.append(num_inputs[2])
        self.block_down1 = nn.Sequential(
            DenseBlock(block_cfg[0][0], num_inputs[0], growth_rate, 4, with_cc=False, mul_dilate=0, layer_type='2d'),
            TransitionDown(num_inputs[1], num_inputs[2], drop_rate, maxpool=False))
        num_inputs = [num_inputs[2], num_inputs[2] + growth_rate * block_cfg[0][1],
                      (num_inputs[2] + growth_rate * block_cfg[0][1]) // 2]
        num_outputs.append(num_inputs[2])
        self.block_down2 = nn.Sequential(
            DenseBlock(block_cfg[0][1], num_inputs[0], growth_rate, 4, with_cc=False, mul_dilate=0, layer_type='2d'),
            TransitionDown(num_inputs[1], num_inputs[2], drop_rate, maxpool=False))
        num_inputs = [num_inputs[2], num_inputs[2] + growth_rate * block_cfg[0][2],
                      (num_inputs[2] + growth_rate * block_cfg[0][2]) // 2]
        self.block_down3 = nn.Sequential(
            DenseBlock(block_cfg[0][2], num_inputs[0], growth_rate, 4, with_cc=False, mul_dilate=1, layer_type='2d'),
            TransitionDown(num_inputs[1], num_inputs[2], drop_rate, maxpool=False))

        # upstream
        num_inputs = [num_inputs[2], num_inputs[2] + growth_rate * block_cfg[1][0],
                      (num_inputs[2] + growth_rate * block_cfg[1][0]) // 2]
        self.block_up1 = nn.Sequential(
            DenseBlock(block_cfg[1][0], num_inputs[0], growth_rate, 4, with_cc=False, mul_dilate=1, layer_type='2d'),
            TransitionUp(num_inputs[1], num_inputs[2], drop=0, deconv=True))
        num_inputs = [num_inputs[2] + num_outputs[1],
                      num_inputs[2] + num_outputs[1] + growth_rate * block_cfg[1][1],
                      (num_inputs[2] + num_outputs[1] + growth_rate * block_cfg[1][1]) // 2]
        self.block_up2 = nn.Sequential(
            DenseBlock(block_cfg[1][1], num_inputs[0], growth_rate, 4, with_cc=False, mul_dilate=1, layer_type='2d'),
            TransitionUp(num_inputs[1], num_inputs[2], drop=0, deconv=True))
        num_inputs = [num_inputs[2] + num_outputs[0],
                      num_inputs[2] + num_outputs[0] + growth_rate * block_cfg[1][2]]
        self.block_up3 = DenseBlock(block_cfg[1][2], num_inputs[0], growth_rate, 4, with_cc=False, mul_dilate=1,
                                    layer_type='2d')

        # outstream
        self.last_conv = nn.Conv2d(num_inputs[1], num_keypoint, 3, 1, 1)
        self.kper = kper

    def forward(self, x):
        if self.inference:
            xl, xr = x

            d1l = self.block_down1(xl)
            d2l = self.block_down2(d1l)
            u1l = self.block_up1(self.block_down3(d2l))
            u2l = self.block_up2(torch.cat((u1l, d2l), dim=1))
            kpl = self.kper(self.last_conv(self.block_up3(torch.cat((u2l, d1l), dim=1))))

            d1r = self.block_down1(xr)
            d2r = self.block_down2(d1r)
            u1r = self.block_up1(self.block_down3(d2r))
            u2r = self.block_up2(torch.cat((u1r, d2r), dim=1))
            kpr = self.kper(self.last_conv(self.block_up3(torch.cat((u2r, d1r), dim=1))))

            return torch.cat((kpl[0], kpr[0]), dim=1)

        else:
            d1 = self.block_down1(x)
            d2 = self.block_down2(d1)
            u1 = self.block_up1(self.block_down3(d2))
            u2 = self.block_up2(torch.cat((u1, d2), dim=1))
            return self.kper(self.last_conv(self.block_up3(torch.cat((u2, d1), dim=1))))


class Decoder(nn.Module):
    def __init__(self, num_keypoint, growth_rate, block_cfg, num_outputs):
        super(Decoder, self).__init__()

        self.model = DenseNetTranspose(num_keypoint, growth_rate, block_cfg, last_transit=False,
                                       with_cc=[0, 1], with_dilate=[0, 1], deconv=True)
        self.outputs = nn.ModuleList()
        for out in num_outputs:
            self.outputs.append(nn.Conv2d(self.model.c_output, out, 3, padding=1))

    def forward(self, x):
        x = self.model(x)
        return [output(x) for output in self.outputs]


class ConverterServo(nn.Module):
    def __init__(self, num_input, growth_rate, block_cfg, num_outputs):
        super(ConverterServo, self).__init__()

        self.model = DenseNet(num_input, growth_rate, block_cfg, layer_type='fc')
        self.outputs = nn.ModuleList()
        for out in num_outputs:
            self.outputs.append(nn.Linear(self.model.c_output, out))

    def forward(self, x):
        x = self.model(x)
        return [output(x).squeeze() for output in self.outputs]


class KeyPointGaussian(nn.Module):
    def __init__(self, sigma, chw):
        super(KeyPointGaussian, self).__init__()
        self.sigma = sigma
        self.c, self.h, self.w = chw

    def forward(self, x):
        n = x.size(0)
        linh = torch.linspace(0, self.h - 1, self.h).cuda().view(1, 1, self.h, 1).expand([1, 1, self.h, self.w])
        linw = torch.linspace(0, self.w - 1, self.w).cuda().view(1, 1, 1, self.w).expand([1, 1, self.h, self.w])
        if x.dim() == 4:
            cmax = F.softmax(x.view(n, self.c, -1), dim=-1).view_as(x)
            cmag = x.view(n, self.c, -1).max(dim=-1)[0].sigmoid().view(n, self.c, 1, 1)
            ctrh = torch.sum((linh*cmax).view(n, self.c, -1), dim=-1).view(n, self.c, 1, 1)
            ctrw = torch.sum((linw*cmax).view(n, self.c, -1), dim=-1).view(n, self.c, 1, 1)
        elif x.dim() == 2:
            cmag = x[:, -self.c:].view(n, self.c, 1, 1)
            ctrh = (x[:, :self.c].view(n, self.c, 1, 1) + 1) * self.h / 2
            ctrw = (x[:, self.c:(2 * self.c)].view(n, self.c, 1, 1) + 1) * self.w / 2
        gaus = torch.exp(-self.sigma * torch.pow(linh - ctrh, 2)) * \
               torch.exp(-self.sigma * torch.pow(linw - ctrw, 2))
        return torch.cat([ctrh * 2 / self.h - 1,
                          ctrw * 2 / self.w - 1,
                          cmag], dim=1).view(n, 3 * self.c), \
               gaus * cmag
