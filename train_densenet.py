import torch
import torch.nn as nn


class Interpolate(nn.Module):
    def __init__(self, scale_factor, mode):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        return self.interp(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=False)


class DenselayerLinear(nn.Module):
    def __init__(self, num_input, growth_rate):
        super(DenselayerLinear, self).__init__()

        self.feature = nn.Sequential()
        self.feature.add_module('fc', nn.Linear(num_input, growth_rate, bias=False))
        self.feature.add_module('norm', nn.BatchNorm1d(growth_rate))
        self.feature.add_module('relu', nn.SiLU())

    def forward(self, x):
        x_ = self.feature(x)
        return torch.cat((x, x_), dim=1)


class TransitionDownLinear(nn.Sequential):
    def __init__(self, num_input, num_output, drop):
        super(TransitionDownLinear, self).__init__()
        self.add_module('fc', nn.Linear(num_input, num_output, bias=False))
        self.add_module('norm', nn.BatchNorm1d(num_output))
        if drop > 0:
            self.add_module('drop', nn.Dropout(drop, inplace=True))
        self.add_module('relu', nn.SiLU())


class DenseLayer1D(nn.Module):
    def __init__(self, num_input, growth_rate, i_layer, mul_dilate):
        super(DenseLayer1D, self).__init__()

        self.feature = nn.Sequential()
        self.feature.add_module('conv', nn.Conv1d(num_input, growth_rate, kernel_size=3, bias=False,
                                                  padding=(i_layer * mul_dilate) + 1,
                                                  dilation=(i_layer * mul_dilate) + 1))
        self.feature.add_module('norm', nn.BatchNorm1d(growth_rate))
        self.feature.add_module('relu', nn.SiLU())

    def forward(self, x):
        x_ = self.feature(x)
        return torch.cat((x, x_), dim=1)


class TransitionDown1D(nn.Sequential):
    def __init__(self, num_input, num_output, drop, maxpool):
        super(TransitionDown1D, self).__init__()
        self.add_module('conv', nn.Conv1d(num_input, num_output, kernel_size=1, bias=False))
        if maxpool:
            self.add_module('pool', nn.MaxPool1d(kernel_size=4, stride=4))
        else:
            self.add_module('pool', nn.AvgPool1d(kernel_size=4, stride=4))
        self.add_module('norm', nn.BatchNorm1d(num_output))
        if drop > 0:
            self.add_module('drop', nn.Dropout(drop, inplace=True))
        self.add_module('relu', nn.SiLU())


class DenseLayer(nn.Module):
    def __init__(self, num_input, growth_rate, bn_size, i_layer, with_cc, mul_dilate):
        super(DenseLayer, self).__init__()

        self.with_cc = with_cc
        self.feature = nn.Sequential()
        if num_input >= ((bn_size + 1) * growth_rate):
            self.feature.add_module('bn_conv', nn.Conv2d(num_input, bn_size * growth_rate,
                                                         kernel_size=1, bias=False))
            self.feature.add_module('bn_norm', nn.BatchNorm2d(bn_size * growth_rate))
            self.feature.add_module('relu', nn.SiLU())
            num_input = bn_size * growth_rate
        self.feature.add_module('conv', nn.Conv2d(num_input, growth_rate, kernel_size=3, bias=False,
                                                  padding=(i_layer * mul_dilate) + 1,
                                                  dilation=(i_layer * mul_dilate) + 1))
        self.feature.add_module('norm', nn.BatchNorm2d(growth_rate))
        self.feature.add_module('relu', nn.SiLU())

    def forward(self, x):
        if self.with_cc:
            n, _, x_d, y_d = x.size()
            x = torch.cat((torch.linspace(-1, 1, x_d).cuda().view(1, 1, x_d, 1).expand([n, 1, x_d, y_d]),
                           torch.linspace(-1, 1, y_d).cuda().view(1, 1, 1, y_d).expand([n, 1, x_d, y_d]),
                           x), dim=1)
        x_ = self.feature(x)
        return torch.cat((x, x_), dim=1)


class DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input, growth_rate, bn_size, with_cc, mul_dilate, layer_type):
        super(DenseBlock, self).__init__()
        num_input += 2 if with_cc else 0
        for i in range(num_layers):
            if layer_type == '1d':
                layer = DenseLayer1D(num_input + i * growth_rate, growth_rate, i, mul_dilate)
            elif layer_type == '2d':
                layer = DenseLayer(num_input + i * growth_rate, growth_rate, bn_size, i, with_cc and i == 0, mul_dilate)
            elif layer_type == 'fc':
                layer = DenselayerLinear(num_input + i * growth_rate, growth_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class TransitionDown(nn.Sequential):
    def __init__(self, num_input, num_output, drop, maxpool):
        super(TransitionDown, self).__init__()
        self.add_module('conv', nn.Conv2d(num_input, num_output, kernel_size=1, bias=False))
        if maxpool:
            self.add_module('pool', nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))
        self.add_module('norm', nn.BatchNorm2d(num_output))
        if drop > 0:
            self.add_module('drop', nn.Dropout(drop, inplace=True))
        self.add_module('relu', nn.SiLU())


class TransitionUp(nn.Sequential):
    def __init__(self, num_input, num_output, drop, deconv):
        super(TransitionUp, self).__init__()
        if deconv:
            self.add_module('deconv', nn.ConvTranspose2d(num_input, num_output, kernel_size=2, stride=2, bias=False))
        else:
            self.add_module('scale', Interpolate(scale_factor=2, mode='bilinear'))
            self.add_module('conv', nn.Conv2d(num_input, num_output, kernel_size=1, bias=False))
        self.add_module('norm', nn.BatchNorm2d(num_output))
        self.add_module('relu', nn.SiLU())
        if drop > 0:
            self.add_module('drop', nn.Dropout(drop, inplace=True))


class DenseNet(nn.Module):
    def __init__(self, input_size, growth_rate, block_cfg,
                 bn_size=4, drop_rate=0, reduction=0.5, with_cc=[], with_dilate=[],
                 mul_dilate=1, last_transit=False, max_pooled=[0], layer_type='2d'):
        super(DenseNet, self).__init__()

        sizes = input_size
        self.features = nn.Sequential()
        for i, num_layers in enumerate(block_cfg):
            block = DenseBlock(num_layers, sizes, growth_rate, bn_size, i in with_cc,
                               mul_dilate if i in with_dilate else 0, layer_type)
            self.features.add_module('denseblock%d' % (i + 1), block)
            sizes += num_layers * growth_rate + (2 if i in with_cc else 0)
            if i < (len(block_cfg) - 1) or last_transit:
                sizes_trans = int(sizes * (1. - reduction))
                if layer_type == '1d':
                    trans = TransitionDown1D(sizes, sizes_trans, drop_rate, i in max_pooled)
                elif layer_type == '2d':
                    trans = TransitionDown(sizes, sizes_trans, drop_rate, i in max_pooled)
                elif layer_type == 'fc':
                    trans = TransitionDownLinear(sizes, sizes_trans, drop_rate)
                self.features.add_module('transitiondown%d' % (i + 1), trans)
                sizes = sizes_trans
        self.c_output = sizes

    def forward(self, x):
        return self.features(x)


class DenseNetTranspose(nn.Module):
    def __init__(self, input_size, growth_rate, block_cfg, reduction=0.5, bn_size=4, drop_rate=0, mul_dilate=1,
                 with_cc=[], with_dilate=[], deconv=True, last_transit=False, layer_type='2d'):
        super(DenseNetTranspose, self).__init__()

        sizes = input_size
        self.features = nn.Sequential()
        for i, num_layers in enumerate(block_cfg):
            block = DenseBlock(num_layers, sizes, growth_rate, bn_size, i in with_cc,
                               mul_dilate if i in with_dilate else 0, layer_type)
            self.features.add_module('denseblock%d' % (i + 1), block)
            sizes += num_layers * growth_rate + (2 if i in with_cc else 0)
            if i < (len(block_cfg) - 1) or last_transit:
                sizes_trans = int(sizes * (1. - reduction))
                trans = TransitionUp(sizes, sizes_trans, drop_rate, deconv)
                self.features.add_module('transitionup%d' % (i + 1), trans)
                sizes = sizes_trans
        self.c_output = sizes

    def forward(self, x):
        return self.features(x)
