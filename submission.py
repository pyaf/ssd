from __future__ import print_function
import pdb
import os
import cv2
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch.backends.cudnn as cudnn
from datetime import datetime
import pydicom
from ssd import build_ssd
from data.dataloader import CLASSES
import torch.utils.data as data


class TestDataset(data.Dataset):
    def __init__(self, root, sample_submission_path):
        self.root = root
        df = pd.read_csv(sample_submission_path)
        self.fnames = list(df['patientId'])
        self.num_samples = len(self.fnames)

    def __getitem__(self, idx):
        pass

    def pull_image(self, idx=0, fname=None):
        if not fname:
            fname = self.fnames[idx]
        dcm_data = pydicom.read_file(self.root + fname + ".dcm")
        img = dcm_data.pixel_array
        img = cv2.resize(img, (300, 300)).astype(np.float32)
        img = np.expand_dims(img, 0).repeat(3, axis=0)
        img = img / 255.
        return img

    def __len__(self):
        return self.num_samples


def get_prediction_str(detections, threshold):
    pred_str = []
    for idx in np.where(detections[0, 1, :, 0] >= threshold)[0]:
        XMin, YMin, XMax, YMax = [x * 1024 for x in detections[0, 1, idx, 1:]]
        score = detections[0, 1, idx, 0]
        w = XMax - XMin + 1
        h = YMax - YMin + 1
        pred_str.extend([score, XMin, YMin, w, h])
    return " ".join(map(lambda x: str(x.item()), pred_str))


if __name__ == "__main__":
    use_cuda = True
    trained_model_path = 'weights/vast/model.pth'
    device = torch.device("cuda" if use_cuda else "cpu")
    if use_cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')
    num_classes = len(CLASSES) + 1  # +1 background
    net = build_ssd('test', 300, num_classes)  # initialize SSD
    state = torch.load(trained_model_path)
    # state = torch.load(trained_model_path, map_location='')
    net.load_state_dict(state["state_dict"])
    net.eval()
    print('Finished loading model!')
    net = net.to(device)
    threshold = get_best_th(net)
    # load data
    root = "data/stage_1_test_images/"
    sample_submission_path = "data/stage_1_sample_submission.csv"
    sub_path = "data/submission/" + str(datetime.now()).replace(" ", "") + '/'
    os.mkdir(sub_path)
    testset = TestDataset(root, sample_submission_path)
    # cudnn.benchmark = True
    num_images = len(testset)
    test_sub = pd.read_csv(sample_submission_path)
    for i in tqdm(range(num_images)):
        img = testset.pull_image(i)
        x = torch.Tensor(img).view(1, 3, 300, 300)
        x = x.to(device)
        y = net(x)      # forward pass
        detections = y.data
        test_sub.at[i, 'PredictionString'] = get_prediction_str(detections, threshold)
    test_sub.to_csv(sub_path + 'submission-%0.2 f.csv' % threshold, index=False)

    '''
    total test images: 2463
    On CPU takes around 1 Hour 40 mins.
    On GPU takes around 1 Hour, 843 MB GPU memory, batch_size=1
    '''