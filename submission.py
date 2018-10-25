from __future__ import print_function
import pdb
import os
import cv2
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch.backends.cudnn as cudnn
import time
import pydicom
from ssd import build_ssd
from dataloader import CLASSES
import torch.utils.data as data


class TestDataset(data.Dataset):
    def __init__(self, root, sample_submission_path, mean=(104, 117, 123)):
        self.root = root
        df = pd.read_csv(sample_submission_path)
        self.fnames = list(df['patientId'])
        self.num_samples = len(self.fnames)
        self.mean = np.array(mean, dtype=np.float32)

    def __getitem__(self, idx):
        pass

    def pull_image(self, idx=0, fname=None):
        if not fname:
            fname = self.fnames[idx]
        dcm_data = pydicom.read_file(self.root + fname + ".dcm")
        img = dcm_data.pixel_array
        img = cv2.resize(img, (300, 300)).astype(np.float32)
        img = np.expand_dims(img, -1).repeat(3, axis=-1)
        img -= self.mean
        img = img.transpose(-1, 0, 1)
        # img /= 255.
        return img

    def __len__(self):
        return self.num_samples


def get_prediction_str(detections, threshold):
    pred_str = []
    for idx in np.where(detections[0, 1, :, 0] >= threshold)[0]:
        XMin, YMin, XMax, YMax = [x * 1024 for x in detections[0, 1, idx, 1:]]
        score = detections[0, 1, idx, 0]
        pred_str.extend([score, XMin, YMin, XMax - XMin, YMax - YMin])
    return " ".join(map(lambda x: str(x.item()), pred_str))


if __name__ == "__main__":
    # load model
    use_cuda = True
    trained_model_path = 'weights/24oct/model.pth'
    print("Using trained model at %s" % trained_model_path)
    device = torch.device("cuda" if use_cuda else "cpu")
    if use_cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')
    num_classes = len(CLASSES) + 1  # +1 background
    net = build_ssd('test', 300, num_classes)  # initialize SSD
    net = torch.nn.DataParallel(net)
    state = torch.load(trained_model_path, map_location=lambda storage, loc: storage)
    net.load_state_dict(state["state_dict"])
    net.eval()
    net = net.to(device)
    print('Finished loading model!')

    # load data
    root = "data/stage_1_test_images/"
    sample_submission_path = "data/stage_1_sample_submission.csv"
    sub_path = "data/submission/" + str(time.time()) + trained_model_path.replace('/', '') + '/'
    os.mkdir(sub_path)
    testset = TestDataset(root, sample_submission_path)

    # cudnn.benchmark = True
    num_images = len(testset)
    test_sub = pd.read_csv(sample_submission_path)
    subs = {}
    thresholds = [0.2, 0.23, 0.27, 0.30, 0.33] # class thresholds for predicted boxes
    print("Using thresholds:", thresholds)
    for i in range(len(thresholds)):
        subs[i] = test_sub.copy()

    # predictions
    for i in tqdm(range(num_images)):
        # pdb.set_trace()
        img = testset.pull_image(i)
        x = torch.Tensor(img).view(1, 3, 300, 300)
        x = x.to(device)
        y = net(x)      # forward pass
        detections = y.data
        for idx, th in enumerate(thresholds):
            subs[idx].at[i, 'PredictionString'] = get_prediction_str(detections, th)

    for idx, th in enumerate(thresholds):
        subs[idx].to_csv(sub_path + 'submission-%0.2f.csv' % th, index=False)


'''
upto vast8oct2 models were trained with img /= 255 normalization, afterwards by img -= mean
'''