import torch
from os.path import dirname, abspath

HOME = dirname(abspath(__file__))
print('HOME:', HOME)

# for making bounding boxes pretty
COLORS = ((255, 0, 0, 128), (0, 255, 0, 128), (0, 0, 255, 128),
          (0, 255, 255, 128), (255, 0, 255, 128), (255, 255, 0, 128))

MEANS = (104, 117, 123)

cuda = torch.cuda.is_available()

cfg = {
    'name': 'RSNA',
    'num_classes': 2,  # 1 is bg
    'lr_steps': (80000, 100000, 120000),
    'max_iter': 120000,
    
    'feature_maps': [38, 19, 10, 5, 3],
    'min_dim': 300,
    'steps': [8, 16, 32, 64, 100],  # distance b/w prior centroids (scaled to input image (300, 300))
    'min_sizes': [60, 90, 120, 150, 180],  # s_k's (scaled to input image) of each feature maps (essentially, w and h)
    'max_sizes': [80, 100, 120, 140, 160],
    'aspect_ratios': [[1/3, 1/2, 2, 3], [1/3, 1/2, 2, 3], [1/3, 1/2, 2, 3], [1/3, 1/2, 2, 3], [1/3, 1/2, 2, 3]],

    'variance': [0.1, 0.2],
    'clip': True,
    'base': {  # used to construct VGG, int: number of out_channels, 'M': MaxPool, 'C': MaxPool with ceil_mode
        '300': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M', 512, 512, 512],
    },
    'extras': {  # used to construct extra layers after VGG and before loc/conf layers
                # int: number of out_channels, 'S': next layer with stride=2 and out_channels as next int
        '300': [256, 'S', 512, 128, 'S', 256, 128, 256],
    },
    'mbox': {
        '300': [5, 5, 5, 5, 5],  # number of boxes per feature map location
    }
}



''' BEFORE 24 oct
cfg = {
    'name': 'RSNA',
    'num_classes': 2,  # 1 is bg
    'lr_steps': (80000, 100000, 120000),
    'max_iter': 120000,
    'feature_maps': [19, 10, 5, 3, 1],
    'min_dim': 300,
    'steps': [16, 32, 64, 100, 300],  # distance b/w prior centroids
    'min_sizes': [60, 80, 100, 120, 140],  # prior widths
    'max_sizes': [80, 100, 120, 140, 160],
    'aspect_ratios': [[2, 3], [2, 3], [2, 3], [2], [2]],
    # 'aspect_ratios': [[0.5, 2], [0.5, 2], [0.5, 2], [0.5, 2], [0.5, 2]],

    'variance': [0.1, 0.2],
    'clip': True,
    'base': {
        '300': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M', 512, 512, 512],
    },
    'extras': {
        '300': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256],
    },
    'mbox': {
        '300': [5, 5, 5, 3, 3],  # number of boxes per feature map location
    }
}
'''


''' cfg used before 23 oct ''' 

# cfg = {
#     'name': 'RSNA',
#     'num_classes': 2,  # 1 is bg
#     'lr_steps': (80000, 100000, 120000),
#     'max_iter': 120000,
#     'feature_maps': [19, 10, 5, 3, 1],
#     'min_dim': 300,
#     'steps': [16, 32, 64, 100, 300],  # distance b/w prior centroids
#     'min_sizes': [60, 80, 100, 120, 140],  # prior widths
#     'max_sizes': [80, 100, 120, 140, 160],
#     # 'aspect_ratios': [[2, 3], [2, 3], [2, 3], [2], [2]],
#     'aspect_ratios': [[0.5, 2], [0.5, 2], [0.5, 2], [0.5, 2], [0.5, 2]],

#     'variance': [0.1, 0.2],
#     'clip': True,
#     'base': {
#         '300': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M', 512, 512, 512],
#     },
#     'extras': {
#         '300': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256],
#     },
#     'mbox': {
#         '300': [1, 1, 1, 1, 1],  # number of boxes per feature map location
#     }
# }




''' orginal cfg '''

# cfg = {
#     'num_classes': 2,  # 1 is bg
#     'lr_steps': (80000, 100000, 120000),
#     'max_iter': 120000,
#     'feature_maps': [38, 19, 10, 5, 3, 1],
#     'min_dim': 300,
#     'steps': [8, 16, 32, 64, 100, 300],
#     'min_sizes': [30, 60, 111, 162, 213, 264],
#     'max_sizes': [60, 111, 162, 213, 264, 315],
#     'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
#     'variance': [0.1, 0.2],
#     'clip': True,
#     'name': 'FACE',
#     'base': {
#         '300': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M', 512, 512, 512],
#     },
#     'extras': {
#         '300': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256],
#     },
#     'mbox': {
#         '300': [4, 6, 6, 6, 4, 4],  # number of boxes per feature map location, 2 for each aspect ratio + 2
#     }
# }

