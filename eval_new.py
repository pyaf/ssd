import matplotlib; matplotlib.use('Agg')
import os
import pdb
import traceback
import torch
from tqdm import tqdm
from ssd import build_ssd
from data.dataloader import CLASSES, provider
from mAP import get_mAP
from extras import get_gt_boxes, get_pred_boxes
from collections import defaultdict

if __name__ == "__main__":
    use_cuda = True
    batch_size = 4
    num_workers = 4
    trained_model_path = 'weights/vast8oct3/model20.pth'
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
    dataloader = provider(phase='val', batch_size=batch_size, num_workers=num_workers)
    running_mAPs = defaultdict(lambda: 0)
    print('Total Iterations: ', len(dataloader))
    predicted_boxes = defaultdict(dict)
    ground_truth_boxes = defaultdict(list)
    try:
        for iteration, batch in tqdm(enumerate(dataloader)):
            fnames, images, targets = batch
            images = images.to(device)
            out = net(images)
            detections = out.data
            for i, name in enumerate(fnames):  # len(images) no batch_size (last iter issue)
                predicted_boxes[name] = get_pred_boxes(detections[i])
                ground_truth_boxes[name] = get_gt_boxes(targets[i])
        print('Predictions ready, calculating mAP...')
        get_mAP(ground_truth_boxes, predicted_boxes)
    except:
        traceback.print_exc()
        pdb.set_trace()