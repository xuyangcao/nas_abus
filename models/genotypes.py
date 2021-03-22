from collections import namedtuple

Genotype = namedtuple('Genotype', 'reduce_cell normal_cell')

PRIMITIVES = [
    'none',
    'max_pool_3x3',
    'avg_pool_3x3',
    'skip_connect',
    'sep_conv_3x3',
    'sep_conv_5x5',
    'dil_conv_3x3',
    'dil_conv_5x5'
]

REDUCE_PRIMITIVES = [
    'conv_3x3',
    'conv_3x3_dil2',
    'conv_3x3_dil3',
    'conv_5x5',
    'conv_5x5_dil2'
]

NORMAL_PRIMITIVES = [
    'conv_1x1',
    'conv_3x3',
    'conv_3x3_dil2',
    'conv_5x5',
    'se_layer',
]


genotype = Genotype(reduce_cell=['conv_5x5_dil2', 'conv_5x5_dil2', 'conv_3x3_dil3', 'conv_3x3', 'conv_3x3_dil3', 'conv_5x5_dil2', 'conv_3x3_dil3', 'conv_3x3_dil3', 'conv_3x3_dil2', 'conv_3x3_dil3', 'conv_3x3_dil3', 'conv_3x3_dil3', 'conv_3x3_dil3', 'conv_3x3', 'conv_3x3', 'conv_3x3', 'conv_5x5_dil2', 'conv_3x3_dil3', 'conv_3x3_dil2', 'conv_3x3_dil3', 'conv_3x3_dil2', 'conv_3x3_dil3', 'conv_3x3_dil3', 'conv_5x5'], normal_cell=['conv_1x1'])

alphas_network = [[0.569934, 0.28881904, 0.14124699],
 [0.57171845, 0.21846177, 0.20981973],
 [0.26953107, 0.5160722,  0.21439673],
 [0.3827924,  0.4162318,  0.20097578],
 [0.25459874, 0.3729547,  0.37244657],
 [0.37064555, 0.24566251, 0.38369203],
 [0.3737104,  0.26425898, 0.36203063],
 [0.4272789,  0.35201842, 0.22070265]]
