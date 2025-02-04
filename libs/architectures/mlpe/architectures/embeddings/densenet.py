import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from torch import Tensor

class _DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate, bn_size, kernel_size, stride=1,  drop_rate=0):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm1d(num_input_features))
        self.add_module('relu1', nn.ReLU(inplace=True))
        self.add_module('conv1', nn.Conv1d(num_input_features, bn_size * growth_rate, kernel_size=1, stride=1, bias=False))
        self.add_module('norm2', nn.BatchNorm1d(bn_size * growth_rate))
        self.add_module('relu2', nn.ReLU(inplace=True)), 
        self.add_module('conv2', nn.Conv1d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False))
        self.drop_rate = float(drop_rate)

    def forward(self, input):
        if isinstance(input, Tensor):
            prev_features = [input]
        else:
            prev_features = input
        concated_features = torch.cat(input, 1)
        bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))
        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return new_features

class _DenseBlock(nn.ModuleDict):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, kernel_size, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                kernel_size=kernel_size,
                drop_rate=drop_rate,
            )
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, init_features: Tensor) -> Tensor:
        features = [init_features]
        for name, layer in self.items():
            new_features = layer(features)
            features.append(new_features)
        return torch.cat(features, 1)

class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm1d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv1d(num_input_features, num_output_features, kernel_size=1, stride=2, bias=False))
        self.add_module('pool', nn.AvgPool1d(kernel_size=2, stride=2))

class DenseNet1D(nn.Module):
    def __init__(self, num_ifos, num_classes, growth_rate=32, block_config=(6, 12, 24, 16), bn_size=4, kernel_size=3, drop_rate=0):
        super(DenseNet1D, self).__init__()

        num_features = 2 * growth_rate
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv1d(num_ifos, num_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm1d(num_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool1d(kernel_size=3, stride=2, padding=1)),
        ]))

        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                kernel_size=kernel_size,
            )
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        self.features.add_module('norm5', nn.BatchNorm1d(num_features))
        self.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.avg_pool1d(out, kernel_size=out.size()[-1]).view(features.size(0), -1)
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out
