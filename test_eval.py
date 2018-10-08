import pdb, traceback
import os
import torch
import numpy as np
from tqdm import tqdm
from ssd import build_ssd
from data.dataloader import CLASSES, provider
from utils import get_mAP
from collections import defaultdict

if __name__ == "__main__":
    use_cuda = True
    batch_size = 12
    trained_model_path = 'weights/vast8oct2/model10.pth'
    device = torch.device("cuda" if use_cuda else "cpu")
    if use_cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')
    state = torch.load(trained_model_path, map_location=lambda storage, loc: storage)
    num_classes = len(CLASSES) + 1  # +1 background
    net = build_ssd('test', 300, num_classes)  # initialize SSD
    net.load_state_dict(state["state_dict"])
    net.eval()
    net = net.to(device)
    print('Finished loading model!')
    dataloader = provider(phase='val', batch_size=batch_size)
    running_mAPs = defaultdict(lambda: 0)
    cls_thresholds = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55]
    print('Total Iterations: ', len(dataloader))
    try:
        for iteration, batch in tqdm(enumerate(dataloader)):
            fnames, images, targets = batch
            images = images.to(device)
            out = net(images)
            detections = out.data
            for i in range(len(images)):  # len(images) no batch_size (last iter issue)
                for cls_th in cls_thresholds:
                    mAP = get_mAP(detections[i], targets[i], cls_th)
                    if mAP is not None:
                        running_mAPs[cls_th] += mAP

        print('mAPs at different class score thresholds are: ')
        for cls_th in cls_thresholds:
            print(cls_th, ': ', running_mAPs[cls_th] / (len(dataloader) * batch_size))
    except:
        traceback.print_exc()
        pdb.set_trace()