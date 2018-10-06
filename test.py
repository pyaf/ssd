from __future__ import print_function
import pdb
import sys
import os
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.autograd import Variable
from data import VOC_ROOT, VOC_CLASSES as labelmap
from PIL import Image
from ssd import *
from data import VOCAnnotationTransform, VOCDetection, BaseTransform, VOC_CLASSES
import torch.utils.data as data
from test_submission import *

cuda=False
if cuda and torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

save_folder = 'eval/'
if not os.path.exists(save_folder):
    os.mkdir(save_folder)

visual_threshold = 0.6
trained_model_path = 'weights/ssd300_COCO_21000.pth'

def test_net(save_folder, net, cuda, testset, transform, thresh):
    # dump predictions and assoc. ground truth to text file for now
    filename = save_folder+'test1.txt'
    num_images = len(testset)
    for i in range(num_images):
        print('Testing image {:d}/{:d}....'.format(i+1, num_images))
        pdb.set_trace()
        img = testset.pull_image(i)
        x = torch.from_numpy(transform(img)[0]).permute(2, 0, 1)
        x = Variable(x.unsqueeze(0))
        if cuda:
            x = x.cuda()

        y = net(x)      # forward pass
        detections = y.data
        # scale each detection back up to the image
        scale = torch.Tensor([img.shape[1], img.shape[0],
                             img.shape[1], img.shape[0]])
        pred_num = 0
        for i in range(detections.size(1)):
            j = 0
            while detections[0, i, j, 0] >= 0.6:
                if pred_num == 0:
                    with open(filename, mode='a') as f:
                        f.write('PREDICTIONS: '+'\n')
                score = detections[0, i, j, 0]
                label_name = labelmap[i-1]
                pt = (detections[0, i, j, 1:]*scale).cpu().numpy()
                coords = (pt[0], pt[1], pt[2], pt[3])
                pred_num += 1
                with open(filename, mode='a') as f:
                    f.write(str(pred_num)+' label: '+label_name+' score: ' +
                            str(score) + ' '+' || '.join(str(c) for c in coords) + '\n')
                j += 1


# load net
num_classes = len(VOC_CLASSES) + 1 # +1 background
net = build_ssd('test', 300, num_classes) # initialize SSD
if not cuda:
	net.load_state_dict(torch.load(trained_model_path,  map_location=lambda storage, loc: storage))
else:
	net.load_state_dict(torch.load(trained_model_path))

net.eval()
print('Finished loading model!')


# load data
root = "../data/image_data/"
sample_submission_path = "../data/sample_submission_fChOj3V.csv"
testset = ListDataset(
    root,
    sample_submission_path,
)

if cuda:
    net = net.cuda()
    cudnn.benchmark = True
# evaluation
test_net(save_folder, net, cuda, testset,
         BaseTransform(net.size, (104, 117, 123)),
         thresh=visual_threshold)