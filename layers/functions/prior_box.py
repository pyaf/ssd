from __future__ import division
import pdb
from math import sqrt as sqrt
from itertools import product as product
import torch
from data.config import cfg


class PriorBox(object):
    """Compute priorbox coordinates in center-offset form for each source
    feature map.
    """
    def __init__(self, cfg):
        super(PriorBox, self).__init__()
        self.image_size = cfg['min_dim']
        # number of priors for feature map location (either 4 or 6)
        self.num_priors = len(cfg['aspect_ratios'])
        self.variance = cfg['variance'] or [0.1]
        self.feature_maps = cfg['feature_maps']
        self.min_sizes = cfg['min_sizes']
        self.max_sizes = cfg['max_sizes']
        self.steps = cfg['steps']
        self.aspect_ratios = cfg['aspect_ratios']
        self.clip = cfg['clip']
        self.version = cfg['name']
        for v in self.variance:
            if v <= 0:
                raise ValueError('Variances must be greater than 0')

    def forward(self):
        mean = []
        # pdb.set_trace()
        for k, f in enumerate(self.feature_maps):
            for i, j in product(range(f), repeat=2):  # all possible coords in fxf feature map
                # f * f priors are there for each fm
                # 389, 100, 25, 3, 1 => prior boxes per fm
                f_k = self.image_size / self.steps[k]
                # unit center x,y
                cx = (j + 0.5) / f_k
                cy = (i + 0.5) / f_k

                # aspect_ratio: 1
                # rel size: min_size
                s_k = self.min_sizes[k] / self.image_size
                mean += [cx, cy, s_k, s_k]
                # print([x*300 for x in mean[-4:]])
                # if s_k * 300 < 50:
                #     pdb.set_trace()

                # aspect_ratio: 1
                # rel size: sqrt(s_k * s_(k+1))
                # s_k_prime = sqrt(s_k * (self.max_sizes[k] / self.image_size))
                # mean += [cx, cy, s_k_prime, s_k_prime]
                # print(mean[-1])

                # rest of aspect ratios
                # for ar in self.aspect_ratios[k]:
                #     mean += [cx, cy, s_k*sqrt(ar), s_k/sqrt(ar)]
                #     # print(mean[-1])
                #     mean += [cx, cy, s_k/sqrt(ar), s_k*sqrt(ar)]
                # print(mean[-1])
        # back to torch land
        output = torch.Tensor(mean).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output


if __name__ == '__main__':
    priors = PriorBox(cfg).forward() * 300
    # for p in priors:
    #     print(p)
    # pdb.set_trace()
