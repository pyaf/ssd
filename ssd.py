import pdb
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn
from layers import PriorBox, L2Norm, Detect
from data.config import cfg
import os


class SSD(nn.Module):
    """Single Shot Multibox Architecture
    The network is composed of a base VGG network followed by the
    added multibox conv layers.  Each multibox layer branches into
        1) conv2d for class conf scores
        2) conv2d for localization predictions
        3) associated priorbox layer to produce default bounding
           boxes specific to the layer's feature map size.
    See: https://arxiv.org/pdf/1512.02325.pdf for more details.

    Args:
        phase: (string) Can be "test" or "train"
        size: input image size
        base: VGG16 layers for input, size of either 300 or 500
        extras: extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
    """

    def __init__(self, phase, size, base, extras, head, num_classes):
        super(SSD, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        self.cfg = cfg
        self.priorbox = PriorBox(self.cfg)
        self.priors = self.priorbox.forward()
        self.size = size

        # SSD network
        self.vgg = nn.ModuleList(base)
        # Layer learns to scale the l2 normalized features from conv4_3
        self.L2Norm = L2Norm(512, 20)
        self.extras = nn.ModuleList(extras)

        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])

        if phase == 'test':
            self.softmax = nn.Softmax(dim=-1)
            self.top_k = 10  # 200 by default
            self.CLS_THRESH = 0.2  # 0.01 by default
            self.NMS_THRESH = 0.3  # earlier had 0.45
            self.detect = Detect(num_classes, self.top_k, self.CLS_THRESH, self.NMS_THRESH)

    def forward(self, x):
        """Applies network layers and ops on input image(s) x.

        Args:
            x: input image or batch of images. Shape: [batch,in_channels,300,300].

        Return:
            Depending on phase:
            test:
                Variable(tensor) of output class label predictions,
                confidence score, and corresponding location predictions for
                each object detected. Shape: [batch,topk,7]

            train:
                list of concat outputs from:
                    1: confidence layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]
                    3: priorbox layers, Shape: [2,num_priors*4]
        """
        sources = list()
        loc = list()
        conf = list()

        # apply vgg up to conv4_3 relu
        for k in range(23):
            x = self.vgg[k](x)

        # s = self.L2Norm(x)
        # sources.append(s)

        # apply vgg up to fc7 (conv7)
        for k in range(23, len(self.vgg)):
            x = self.vgg[k](x)
        sources.append(x)

        # apply extra layers and cache source layer outputs
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
                sources.append(x)

        # apply multibox head to source layers
        for (x, l, c) in zip(sources, self.loc, self.conf):
            # print(np.prod(l(x).shape) // 4)
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())
            # permuted so that all elems in could be flattened and concatenated
        # prod(l(x).shape)//4 = 5776, 2166, 600, 150, 36, 4 => 8732
        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)  # [1, 34928]
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)  # [1, 17464]
        # pdb.set_trace()
        if self.phase == "test":
            output = self.detect(
                loc.view(loc.size(0), -1, 4),                   # [1, 8732, 4]
                self.softmax(conf.view(conf.size(0), -1,
                                       self.num_classes)),      # [1, 8732, num_classes]
                self.priors.type(type(x.data))                  # default boxes
            )
        else:
            output = [
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
                self.priors
            ]

        return output

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file,
                                            map_location=lambda storage, loc: storage))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')


# This function is derived from torchvision VGG make_layers()
# https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
def vgg(cfg, in_channels, batch_norm=False):
    layers = []
    for v in cfg:
        if v == 'M':
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        elif v == 'C':
            layers.append(nn.MaxPool2d(
                kernel_size=2, stride=2, ceil_mode=True))
        else:
            layers.append(nn.Conv2d(in_channels, v, kernel_size=3, padding=1))
            if batch_norm:
                layers.append(nn.BatchNorm2d(v))
            layers.append(nn.ReLU(inplace=True))
            in_channels = v
    layers.append(nn.MaxPool2d(kernel_size=3, stride=1, padding=1))  # pool5
    layers.append(nn.Conv2d(512, 1024, kernel_size=3,
                            padding=6, dilation=6))  # conv6
    layers.append(nn.ReLU(inplace=True))
    layers.append(nn.Conv2d(1024, 1024, kernel_size=1))  # conv7
    layers.append(nn.ReLU(inplace=True))
    return layers


def add_extras(cfg, in_channels, batch_norm=False):
    # Extra layers added to VGG for feature scaling
    layers = []
    flag = False
    for k, v in enumerate(cfg):
        if in_channels != 'S':
            if v == 'S':
                layers.append(nn.Conv2d(
                    in_channels, cfg[k + 1], kernel_size=(1, 3)[flag], stride=2, padding=1))
            else:
                layers.append(nn.Conv2d(in_channels, v,
                                        kernel_size=(1, 3)[flag]))
            flag = not flag
        in_channels = v
    return layers


def multibox(vgg, extra_layers, mbox, num_classes):
    loc_layers = []
    conf_layers = []
    # vgg_source = [21, -2]  # conv4_3 and conv7 of vgg
    vgg_source = [-2]  # conv4_3 and conv7 of vgg
    for k, v in enumerate(vgg_source):
        loc_layers += [nn.Conv2d(vgg[v].out_channels,
                                 mbox[k] * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(vgg[v].out_channels,
                                  mbox[k] * num_classes, kernel_size=3, padding=1)]
    for k, v in enumerate(extra_layers[1::2], len(vgg_source)):  # k starts with 2, v = [512, 256, 256, 256]
        loc_layers += [nn.Conv2d(v.out_channels, mbox[k]
                                 * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(v.out_channels, mbox[k]
                                  * num_classes, kernel_size=3, padding=1)]
    return vgg, extra_layers, (loc_layers, conf_layers)


def build_ssd(phase, size=300, num_classes=2):
    if phase != "test" and phase != "train":
        print("ERROR: Phase: " + phase + " not recognized")
        return
    if size != 300:
        print("ERROR: You specified size " + repr(size) + ". However, " +
              "currently only SSD300 (size=300) is supported!")
        return
    in_channels = 3  # for rsna competition image is single channeled.
    base = cfg['base']
    extras = cfg['extras']
    mbox = cfg['mbox']
    base_, extras_, head_ = multibox(vgg(base[str(size)], in_channels),
                                     add_extras(extras[str(size)], 1024),
                                     mbox[str(size)], num_classes)
    return SSD(phase, size, base_, extras_, head_, num_classes)


if __name__ == '__main__':
    '''If using nn.DataParallel make sure the batch size is greater than num of GPUs.'''
    net = build_ssd("train", cfg["min_dim"], cfg["num_classes"])
    use_gpu = True
    device = torch.device("cuda:0" if use_gpu else "cpu")
    net = torch.nn.DataParallel(net)
    net = net.to(device)
    test_input = torch.rand(8, 3, 300, 300).to(device)
    pdb.set_trace()
    loc, conf, priors = net(test_input)
    print(loc.shape, conf.shape, priors.shape)
