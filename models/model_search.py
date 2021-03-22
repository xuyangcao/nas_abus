import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3' 
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from models.genotypes import REDUCE_PRIMITIVES, NORMAL_PRIMITIVES, Genotype
from models.operations import *
#from genotypes import REDUCE_PRIMITIVES, NORMAL_PRIMITIVES, Genotype
#from operations import *


class ReduceMixedOp(nn.Module):
    def __init__(self, C_in, C_out):
        super(ReduceMixedOp, self).__init__()

        self._ops = nn.ModuleList()
        for primitive in REDUCE_PRIMITIVES:
            op = OPS[primitive](C_in, C_out)
            self._ops.append(op)

    def forward(self, x, weights):
        return sum(w * op(x) for w, op in zip(weights, self._ops))


class NormalMixedOp(nn.Module):
    def __init__(self, C_in, C_out):
        super(NormalMixedOp, self).__init__()

        self._ops = nn.ModuleList()
        for primitive in NORMAL_PRIMITIVES:
            op = OPS[primitive](C_in, C_out)
            self._ops.append(op)

    def forward(self, x, weights):
        return sum(w * op(x) for w, op in zip(weights, self._ops))


class DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, padding, dilation):
        super(DenseLayer, self).__init__()
        # modules for bottle neck layer
        self.add_module('norm1', nn.BatchNorm3d(num_input_features))
        self.add_module('relu1', nn.ReLU(inplace=True))
        self.add_module('conv1', nn.Conv3d(num_input_features, bn_size * growth_rate, 
                                            kernel_size=1, stride=1, bias=False))
        # modules for dense layer
        self.op = ReduceMixedOp(bn_size*growth_rate, growth_rate) # input, output
        self.drop_rate = float(drop_rate)

    def bn_function(self, inputs):
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))
        return bottleneck_output

    def forward(self, inputs, weights):
        prev_features = inputs
        bottleneck_output = self.bn_function(prev_features)
        new_features = self.op(bottleneck_output, weights)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return new_features


class ReduceCell(nn.Module):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
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
                    dilation=1
                    )
            self.layers.append(layer)

        num_output_features = num_input_features + num_layers * growth_rate
        self.reduce = ReduceLayer(num_output_features, num_output_features // 2)

    def forward(self, init_features, weights):
        features = [init_features]
        for i, layer in enumerate(self.layers):
            new_features= layer(features, weights[i])
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
    def __init__(self, in_planes, out_planes, rate):
        super(NormalCell, self).__init__()

        if rate == 2:
            self.preprocess = ReduceLayer(in_planes, out_planes)
        elif rate == 0:
            self.preprocess = ExpandLayer(in_planes, out_planes)
        else:
            self.preprocess = ReLUConvBN(in_planes, out_planes, 1, 1, 0)

        self.bn = nn.BatchNorm3d(out_planes)
        self.relu = nn.ReLU(inplace=True)
        self.op = NormalMixedOp(out_planes, out_planes)

    def forward(self, x, weights):
        x = self.preprocess(x)
        #x = self.op(x, weights[0])
        x = self.op(self.relu(self.bn(x)), weights[0])
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
    def __init__(self, in_channels=3, num_classes=2, criterion=None, growth_rate=16, num_init_features=32, bn_size=4, drop_rate=0.3, block_config=(6, 12, 12, 6)):
        super(UXNet, self).__init__()
        self.criterion = criterion
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.growth_rate = growth_rate
        self.num_init_features = num_init_features
        self.bn_size = bn_size
        self.drop_rate = drop_rate
        self.block_config = block_config

        self._criterion = criterion
        self._initialize_alphas()

        self.stem = nn.Sequential(
            nn.Conv3d(self.in_channels, num_init_features, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm3d(num_init_features),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
            )

        # LayerName: layer_level_next.layer
        num_layers = block_config[0]
        num_features = num_init_features
        self.cell_00 = ReduceCell(num_layers, num_features, bn_size, growth_rate, drop_rate) 
        num_features_d0 = (num_features + num_layers * growth_rate) // 2

        num_layers = block_config[1]
        self.cell_11 = ReduceCell(num_layers, num_features_d0, bn_size, growth_rate, drop_rate) 
        num_features_d1 = (num_features_d0 + num_layers * growth_rate) // 2

        num_layers = block_config[2]
        self.cell_22 = ReduceCell(num_layers, num_features_d1, bn_size, growth_rate, drop_rate) 
        num_features_d2 = (num_features_d1 + num_layers * growth_rate) // 2

        num_layers = block_config[3]
        self.cell_33 = ReduceCell(num_layers, num_features_d2, bn_size, growth_rate, drop_rate)
        num_features_d3 = (num_features_d2 + num_layers * growth_rate) // 2

        self.cell_000 = NormalCell(num_features_d0, num_features_d0, rate=1) 
        self.cell_001 = NormalCell(num_features_d0, num_features_d0, rate=1) 
        self.cell_002 = NormalCell(num_features_d0, num_features_d1, rate=2) 

        self.cell_110 = NormalCell(num_features_d1, num_features_d0, rate=0) 
        self.cell_111 = NormalCell(num_features_d1, num_features_d1, rate=1) 
        self.cell_112 = NormalCell(num_features_d1, num_features_d1, rate=1) 

        self.cell_220 = NormalCell(num_features_d2, num_features_d1, rate=0) 
        self.cell_221 = NormalCell(num_features_d2, num_features_d2, rate=1) 
        self.cell_222 = NormalCell(num_features_d2, num_features_d2, rate=1) 

        self.cell_330 = NormalCell(num_features_d3, num_features_d2, rate=0) 
        self.cell_53 = NormalCell(num_features_d3, num_features_d3, rate=1) 

        self.cell_310 = NormalCell(num_features_d1, num_features_d0, rate=0) 
        self.cell_311 = NormalCell(num_features_d1, num_features_d1, rate=1) 
        self.cell_312 = NormalCell(num_features_d1, num_features_d2, rate=2) 

        self.cell_400 = NormalCell(num_features_d0, num_features_d0, rate=1) 
        self.cell_401 = NormalCell(num_features_d0, num_features_d1, rate=2) 
        self.cell_402 = NormalCell(num_features_d0, num_features_d1, rate=2) 

        self.cell_420 = NormalCell(num_features_d2, num_features_d1, rate=0) 
        self.cell_421 = NormalCell(num_features_d2, num_features_d2, rate=1) 
        self.cell_422 = NormalCell(num_features_d2, num_features_d3, rate=2) 

        self.cell_510 = NormalCell(num_features_d1, num_features_d0, rate=0) 
        self.cell_511 = NormalCell(num_features_d1, num_features_d1, rate=1) 
        self.cell_512 = NormalCell(num_features_d1, num_features_d2, rate=2) 

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

    def new(self):
        model_new = UXNet(self.in_channels, self.num_classes, self.criterion, self.growth_rate, self.num_init_features, self.bn_size, self.drop_rate, self.block_config).cuda()

        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)
        return model_new

    def forward(self, x):
        # we only softmax once, because weight network is different for each cell
        weight_network = F.softmax(self.alphas_network, dim=-1) 

        x = self.stem(x)

        # softmax alphas_reduce_cell only once
        weights_reduce = F.softmax(self.alphas_reduce_cell, dim=-1)
        x_00 = self.cell_00(x, weights_reduce[0:self.block_config[0]])
        x_11 = self.cell_11(x_00, weights_reduce[self.block_config[0]:sum(self.block_config[0:2])])
        x_22 = self.cell_22(x_11, weights_reduce[sum(self.block_config[0:2]):sum(self.block_config[0:3])])
        x_33 = self.cell_33(x_22, weights_reduce[sum(self.block_config[0:3]):sum(self.block_config[0:4])])
        #print('x00.shape: ', x_00.shape)
        #print('x11.shape: ', x_11.shape)
        #print('x22.shape: ', x_22.shape)
        #print('x33.shape: ', x_33.shape)

        # 3,1 softmax alphas_normal_cell every time
        weights_normal = F.softmax(self.alphas_normal_cell, dim=-1)
        input_0 = self.cell_002(x_00, weights_normal)
        weights_normal = F.softmax(self.alphas_normal_cell, dim=-1)
        input_1 = self.cell_112(x_11, weights_normal)
        weights_normal = F.softmax(self.alphas_normal_cell, dim=-1)
        input_2 = self.cell_220(x_22, weights_normal)
        x_31 = weight_network[0][0] * input_0 + weight_network[0][1] * input_1 + weight_network[0][2] * input_2 

        # 4, 0
        weights_normal = F.softmax(self.alphas_normal_cell, dim=-1)
        input_0 = self.cell_001(x_00, weights_normal)
        weights_normal = F.softmax(self.alphas_normal_cell, dim=-1)
        input_1 = self.cell_110(x_11, weights_normal)
        weights_normal = F.softmax(self.alphas_normal_cell, dim=-1)
        input_2 = self.cell_310(x_31, weights_normal)
        x_40 = weight_network[1][0] * input_0 + weight_network[1][1] * input_1 + weight_network[0][2] * input_2 

        # 4, 2
        weights_normal = F.softmax(self.alphas_normal_cell, dim=-1)
        input_0 = self.cell_312(x_31, weights_normal)
        weights_normal = F.softmax(self.alphas_normal_cell, dim=-1)
        input_1 = self.cell_222(x_22, weights_normal)
        weights_normal = F.softmax(self.alphas_normal_cell, dim=-1)
        input_2 = self.cell_330(x_33, weights_normal)
        x_42 = weight_network[2][0] * input_0 + weight_network[2][1] * input_1 + weight_network[2][2] * input_2

        # 5, 1
        weights_normal = F.softmax(self.alphas_normal_cell, dim=-1)
        input_0 = self.cell_402(x_40, weights_normal)
        weights_normal = F.softmax(self.alphas_normal_cell, dim=-1)
        input_1 = self.cell_311(x_31, weights_normal)
        weights_normal = F.softmax(self.alphas_normal_cell, dim=-1)
        input_2 = self.cell_420(x_42, weights_normal)
        x_51 = weight_network[3][0] * input_0 + weight_network[3][1] * input_1 + weight_network[3][2] * input_2

        # 5, 3
        weights_normal = F.softmax(self.alphas_normal_cell, dim=-1)
        input_0 = self.cell_422(x_42, weights_normal)
        weights_normal = F.softmax(self.alphas_normal_cell, dim=-1)
        input_2 = self.cell_53(x_33, weights_normal)
        x_53 = weight_network[4][0] * input_0 + input_2 
        #del(x_33)

        # 6,2
        weights_normal = F.softmax(self.alphas_normal_cell, dim=-1)
        input_0 = self.cell_512(x_51, weights_normal)
        weights_normal = F.softmax(self.alphas_normal_cell, dim=-1)
        input_1 = self.cell_421(x_42, weights_normal)
        weights_normal = F.softmax(self.alphas_normal_cell, dim=-1)
        input_2 = self.cell_221(x_22, weights_normal)
        input_3 = self.cell_530(x_53)
        x_62 = weight_network[5][0] * input_0 + weight_network[5][1] * input_1 + weight_network[5][2] * input_2 + input_3 
        #del(x_22)

        # 7,1
        weights_normal = F.softmax(self.alphas_normal_cell, dim=-1)
        input_0 = self.cell_401(x_40, weights_normal)
        weights_normal = F.softmax(self.alphas_normal_cell, dim=-1)
        input_1 = self.cell_511(x_51, weights_normal)
        weights_normal = F.softmax(self.alphas_normal_cell, dim=-1)
        input_2 = self.cell_111(x_11, weights_normal)
        input_3 = self.cell_620(x_62)
        x_71 = weight_network[6][0] * input_0 + weight_network[6][1] * input_1 + weight_network[6][2] * input_2 + input_3 
        #del(x_11)

        # 8,0
        weights_normal = F.softmax(self.alphas_normal_cell, dim=-1)
        input_0 = self.cell_000(x_00, weights_normal)
        weights_normal = F.softmax(self.alphas_normal_cell, dim=-1)
        input_1 = self.cell_400(x_40, weights_normal)
        weights_normal = F.softmax(self.alphas_normal_cell, dim=-1)
        input_2 = self.cell_510(x_51, weights_normal)
        input_3 = self.cell_710(x_71)
        x_80 = weight_network[7][0] * input_0 + weight_network[7][1] * input_1 + weight_network[7][2] * input_2 + input_3

        # expand layer
        x_80 = self.cell_80(x_80)

        return self.out(x_80) 

    @property
    def config(self, ):
        return {
                'in_channels': self.in_channels,
                'num_classes': self.num_classes,
                'growth_rate': self.growth_rate,
                'num_init_features': self.num_init_features,
                'bn_size': self.bn_size,
                'drop_rate': self.drop_rate,
                'block_config': self.block_config
                }

    def _initialize_alphas(self, ):
        k = sum(self.block_config)
        num_ops = len(REDUCE_PRIMITIVES)
        self.alphas_reduce_cell = Variable(1e-3*torch.randn(k, num_ops).cuda(), requires_grad=True)

        num_ops = len(NORMAL_PRIMITIVES)
        self.alphas_normal_cell = Variable(1e-3*torch.randn(1, num_ops).cuda(), requires_grad=True)

        self.alphas_network = Variable(1e-3*torch.randn(8, 3).cuda(), requires_grad=True)

        self._arch_parameters = [
                self.alphas_reduce_cell,
                self.alphas_normal_cell,
                self.alphas_network
                ]

    def arch_parameters(self):
        return self._arch_parameters

    def _loss(self, input, target) :
        logits = self(input)
        logits = F.softmax(logits, dim=1)
        return self._criterion(logits[:, 1, ...], target==1)


    def _genotype(self):
        def _parse(weights):
            gene = []
            n = 1 
            start = 0
            for i in range(self._step):
                end = start + n
                W = weights[start:end].copy()
                edges = sorted (range(i + 1), key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != PRIMITIVES.index('none')))[:2]
                print('edges: ', edges)
                for j in edges:
                    k_best = None
                    for k in range(len(W[j])):
                        if k != PRIMITIVES.index('none'):
                            if k_best is None or W[j][k] > W[j][k_best]:
                                k_best = k
                    gene.append((PRIMITIVES[k_best], j))
                start = end
                n += 1
            return gene

        gene_cell = _parse(F.softmax(self.alphas_cell, dim=-1).data.cpu().numpy())
        concat = range(2+self._step-self._multiplier, self._step+2)
        genotype = Genotype(
            cell=gene_cell, cell_concat=concat
        )

        return genotype

    def genotype(self, ):
        def _parse(weights, PRIMITIVES):
            gene = []
            k_best = None 
            for i in range(len(weights)):
                w = weights[i]
                for k in range(len(w)):
                    if k_best is None or w[k] > w[k_best]:
                        k_best = k
                gene.append(PRIMITIVES[k_best])
            return gene
        gene_reduce_cell = _parse(F.softmax(self.alphas_reduce_cell, dim=-1).data.cpu().numpy(), REDUCE_PRIMITIVES)
        gene_normal_cell = _parse(F.softmax(self.alphas_normal_cell, dim=-1).data.cpu().numpy(), NORMAL_PRIMITIVES)
        genotype = Genotype(reduce_cell=gene_reduce_cell, normal_cell=gene_normal_cell)
        return genotype 

if __name__ == "__main__":
    model = UXNet()
    x = torch.ones(5, 3, 128, 512)

    n_params = sum([p.data.nelement() for p in model.parameters()])
    print('--- total parameters = {} ---'.format(n_params))

    model = model.cuda()
    x = x.cuda()

    y = model(x)
    print(y.shape)
    print('genotype: {}'.format(model.genotype()))
