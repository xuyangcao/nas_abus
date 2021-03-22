import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '0' 
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from models.genotypes import REDUCE_PRIMITIVES, NORMAL_PRIMITIVES, Genotype
from models.operations import *
from models.genotypes import genotype, alphas_network  
#from genotypes import REDUCE_PRIMITIVES, NORMAL_PRIMITIVES, Genotype
#from operations import *
#from genotypes import genotype, alphas_network  


class DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, padding, dilation, op_name):
        super(DenseLayer, self).__init__()
        # modules for bottle neck layer
        self.add_module('norm1', nn.BatchNorm3d(num_input_features))
        self.add_module('relu1', nn.ReLU(inplace=True))
        self.add_module('conv1', nn.Conv3d(num_input_features, bn_size * growth_rate, 
                                            kernel_size=1, stride=1, bias=False))
        # modules for dense layer
        self.op = OPS[op_name](bn_size*growth_rate, growth_rate) 
        self.drop_rate = float(drop_rate)

    def bn_function(self, inputs):
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))
        return bottleneck_output

    def forward(self, inputs):
        prev_features = inputs
        bottleneck_output = self.bn_function(prev_features)
        new_features = self.op(bottleneck_output)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return new_features


class ReduceCell(nn.Module):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, op_names):
        super(ReduceCell, self).__init__()
        self.num_layers = num_layers

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer = DenseLayer(
                    num_input_features + i * growth_rate,
                    growth_rate=growth_rate,
                    bn_size=bn_size,
                    drop_rate=drop_rate,
                    padding=1,
                    dilation=1,
                    op_name=op_names[i]
                    )
            self.layers.append(layer)

        num_output_features = num_input_features + num_layers * growth_rate
        self.reduce = ReduceLayer(num_output_features, num_output_features // 2)

    def forward(self, init_features):
        features = [init_features]
        for i, layer in enumerate(self.layers):
            new_features= layer(features)
            features.append(new_features)
        features = torch.cat(features, 1)
        return self.reduce(features) 


class ExpandCell(nn.Module):
    def __init__(self, in_planes, out_planes, scale_factor=(2, 2, 2)):
        super(ExpandCell, self).__init__()

        self.scale_factor = scale_factor
        self.norm = nn.BatchNorm3d(in_planes)
        self.up = nn.functional.interpolate

        self.conv = nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm3d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.norm(x)
        x = self.up(x, scale_factor=self.scale_factor, mode='nearest')
        x = self.relu(self.bn(self.conv(x)))
        return x


class NormalCell(nn.Module):
    def __init__(self, in_planes, out_planes, rate, op_name):
        super(NormalCell, self).__init__()

        if rate == 2:
            self.preprocess = ReduceLayer(in_planes, out_planes)
        elif rate == 0:
            self.preprocess = ExpandLayer(in_planes, out_planes)
        else:
            self.preprocess = ReLUConvBN(in_planes, out_planes, 1, 1, 0)

        self.bn = nn.BatchNorm3d(out_planes)
        self.relu = nn.ReLU(inplace=True)
        self.op = OPS[op_name](out_planes, out_planes) 

    def forward(self, x):
        x = self.preprocess(x)
        #x = self.bn(self.op(self.relu(x)))
        x = self.op(self.relu(self.bn(x)))
        #x = self.op(x)
        return x


class OutputBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutputBlock, self).__init__()
        self.norm = nn.BatchNorm3d(in_channels)
        self.up = nn.functional.interpolate
        self.conv = nn.Conv3d(in_channels, in_channels, 3, padding=1)
        self.bn = nn.BatchNorm3d(in_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv3d(in_channels, in_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm3d(in_channels)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv3d(in_channels, out_channels, 1)

    def forward(self, x):
        x = self.norm(x)
        x = self.up(x, scale_factor=2, mode='trilinear', align_corners=False)
        x = self.relu(self.bn(self.conv(x))) 
        x = self.up(x, scale_factor=2, mode='trilinear', align_corners=False)
        x = self.relu2(self.bn2(self.conv2(x))) 
        x = self.conv3(x)
        return x


class UXNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=2, growth_rate=32, num_init_features=64, bn_size=4, drop_rate=0.3, block_config=(6, 12, 24, 16), threshold=0, genotype=genotype, weight_network=alphas_network):
        super(UXNet, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.growth_rate = growth_rate
        self.num_init_features = num_init_features
        self.bn_size = bn_size
        self.drop_rate = drop_rate
        self.block_config = block_config

        weight_network = np.array(weight_network, dtype=np.float32)
        weight_network = torch.from_numpy(weight_network)
        if threshold is not None:
            weight_network = weight_network > threshold
        self.weight_network = weight_network.float()
        print('weight network: ', self.weight_network)

        self.stem = nn.Sequential(
            nn.Conv3d(self.in_channels, num_init_features, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm3d(num_init_features),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
            )

        op_names_reduce = genotype.reduce_cell
        op_names_normal = genotype.normal_cell

        # LayerName: layer_level_next.layer
        num_layers = block_config[0]
        num_features = num_init_features
        op_names = op_names_reduce[0:self.block_config[0]]
        self.cell_00 = ReduceCell(num_layers, num_features, bn_size, growth_rate, drop_rate, op_names) 
        num_features_d0 = (num_features + num_layers * growth_rate) // 2

        num_layers = block_config[1]
        op_names = op_names_reduce[self.block_config[0]:sum(self.block_config[0:2])]
        self.cell_11 = ReduceCell(num_layers, num_features_d0, bn_size, growth_rate, drop_rate, op_names) 
        num_features_d1 = (num_features_d0 + num_layers * growth_rate) // 2

        num_layers = block_config[2]
        op_names = op_names_reduce[sum(self.block_config[0:2]):sum(self.block_config[0:3])]
        self.cell_22 = ReduceCell(num_layers, num_features_d1, bn_size, growth_rate, drop_rate, op_names) 
        num_features_d2 = (num_features_d1 + num_layers * growth_rate) // 2

        num_layers = block_config[3]
        op_names = op_names_reduce[sum(self.block_config[0:3]):sum(self.block_config[0:4])]
        self.cell_33 = ReduceCell(num_layers, num_features_d2, bn_size, growth_rate, drop_rate, op_names)
        num_features_d3 = (num_features_d2 + num_layers * growth_rate) // 2

        self.cell_000 = NormalCell(num_features_d0, num_features_d0, rate=1, op_name=op_names_normal[0]) 
        self.cell_001 = NormalCell(num_features_d0, num_features_d0, rate=1, op_name=op_names_normal[0]) 
        self.cell_002 = NormalCell(num_features_d0, num_features_d1, rate=2, op_name=op_names_normal[0]) 

        self.cell_110 = NormalCell(num_features_d1, num_features_d0, rate=0, op_name=op_names_normal[0]) 
        self.cell_111 = NormalCell(num_features_d1, num_features_d1, rate=1, op_name=op_names_normal[0]) 
        self.cell_112 = NormalCell(num_features_d1, num_features_d1, rate=1, op_name=op_names_normal[0]) 

        self.cell_220 = NormalCell(num_features_d2, num_features_d1, rate=0, op_name=op_names_normal[0]) 
        self.cell_221 = NormalCell(num_features_d2, num_features_d2, rate=1, op_name=op_names_normal[0]) 
        self.cell_222 = NormalCell(num_features_d2, num_features_d2, rate=1, op_name=op_names_normal[0]) 

        self.cell_330 = NormalCell(num_features_d3, num_features_d2, rate=0, op_name=op_names_normal[0]) 
        self.cell_53 = NormalCell(num_features_d3, num_features_d3, rate=1, op_name=op_names_normal[0]) 

        self.cell_310 = NormalCell(num_features_d1, num_features_d0, rate=0, op_name=op_names_normal[0]) 
        self.cell_311 = NormalCell(num_features_d1, num_features_d1, rate=1, op_name=op_names_normal[0]) 
        self.cell_312 = NormalCell(num_features_d1, num_features_d2, rate=2, op_name=op_names_normal[0]) 

        self.cell_400 = NormalCell(num_features_d0, num_features_d0, rate=1, op_name=op_names_normal[0]) 
        self.cell_401 = NormalCell(num_features_d0, num_features_d1, rate=2, op_name=op_names_normal[0]) 
        self.cell_402 = NormalCell(num_features_d0, num_features_d1, rate=2, op_name=op_names_normal[0]) 

        self.cell_420 = NormalCell(num_features_d2, num_features_d1, rate=0, op_name=op_names_normal[0]) 
        self.cell_421 = NormalCell(num_features_d2, num_features_d2, rate=1, op_name=op_names_normal[0]) 
        self.cell_422 = NormalCell(num_features_d2, num_features_d3, rate=2, op_name=op_names_normal[0]) 

        self.cell_510 = NormalCell(num_features_d1, num_features_d0, rate=0, op_name=op_names_normal[0]) 
        self.cell_511 = NormalCell(num_features_d1, num_features_d1, rate=1, op_name=op_names_normal[0]) 
        self.cell_512 = NormalCell(num_features_d1, num_features_d2, rate=2, op_name=op_names_normal[0]) 

        self.cell_530 = ExpandCell(num_features_d3, num_features_d2) 
        self.cell_620 = ExpandCell(num_features_d2, num_features_d1) 
        self.cell_710 = ExpandCell(num_features_d1, num_features_d0) 
        self.cell_80 = ExpandCell(num_features_d0, num_init_features)

        self.out = OutputBlock(num_init_features, self.num_classes)
        self._init_weights()

    def _init_weights(self, ):
        # weight init
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)

    def forward(self, x):
        weight_network = self.weight_network

        x = self.stem(x)

        x_00 = self.cell_00(x)
        x_11 = self.cell_11(x_00)
        x_22 = self.cell_22(x_11)
        x_33 = self.cell_33(x_22)

        # 3,1
        input_0 = self.cell_002(x_00)
        input_1 = self.cell_112(x_11)
        input_2 = self.cell_220(x_22)
        x_31 = weight_network[0][0] * input_0 + weight_network[0][1] * input_1 + weight_network[0][2] * input_2 

        # 4, 0
        input_0 = self.cell_001(x_00)
        input_1 = self.cell_110(x_11)
        input_2 = self.cell_310(x_31)
        x_40 = weight_network[1][0] * input_0 + weight_network[1][1] * input_1 + weight_network[0][2] * input_2 

        # 4, 2
        input_0 = self.cell_312(x_31)
        input_1 = self.cell_222(x_22)
        input_2 = self.cell_330(x_33)
        x_42 = weight_network[2][0] * input_0 + weight_network[2][1] * input_1 + weight_network[2][2] * input_2

        # 5, 1
        input_0 = self.cell_402(x_40)
        input_1 = self.cell_311(x_31)
        input_2 = self.cell_420(x_42)
        x_51 = weight_network[3][0] * input_0 + weight_network[3][1] * input_1 + weight_network[3][2] * input_2

        # 5, 3
        input_0 = self.cell_422(x_42)
        input_2 = self.cell_53(x_33)
        x_53 = weight_network[4][0] * input_0 + input_2 
        #del(x_33)

        # 6,2
        input_0 = self.cell_512(x_51)
        input_1 = self.cell_421(x_42)
        input_2 = self.cell_221(x_22)
        input_3 = self.cell_530(x_53)
        x_62 = weight_network[5][0] * input_0 + weight_network[5][1] * input_1 + weight_network[5][2] * input_2 + input_3 
        #del(x_22)

        # 7,1
        input_0 = self.cell_401(x_40)
        input_1 = self.cell_511(x_51)
        input_2 = self.cell_111(x_11)
        input_3 = self.cell_620(x_62)
        x_71 = weight_network[6][0] * input_0 + weight_network[6][1] * input_1 + weight_network[6][2] * input_2 + input_3 
        #del(x_11)

        # 8,0
        input_0 = self.cell_000(x_00)
        input_1 = self.cell_400(x_40)
        input_2 = self.cell_510(x_51)
        input_3 = self.cell_710(x_71)
        x_80 = weight_network[7][0] * input_0 + weight_network[7][1] * input_1 + weight_network[7][2] * input_2 + input_3

        # expand layer
        x_80 = self.cell_80(x_80)
        #print('x80.shape', x_80.shape)

        return self.out(x_80) 


if __name__ == "__main__":
    model = UXNet()
    x = torch.ones(1, 3, 128, 512)
    n_params = sum([p.data.nelement() for p in model.parameters()])
    print('--- total parameters = {} ---'.format(n_params))

    model = model.cuda()
    x = x.cuda()

    y = model(x)
    print(x.shape)
    print(y.shape)
