import pydicom
import os
import pdb, traceback
import random

import torch
import torch.utils.data as data
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# from transform import resize, random_flip, random_crop, center_crop
from utils.augmentations import SSDAugmentation
from data.config import HOME, cfg, MEANS

CLASSES = ("Lung Opacity",)  # COMMA IS FUCKING IMPORTANT

# CLASSES = ("Lung Opacity", "No Lung Opacity / Not Normal")
class_to_idx = {
    "Normal": -1,
    "Lung Opacity": 0,
    "No Lung Opacity / Not Normal": 1
}

# as Normal and No Lung Opacity / Not Normal have same Target (0), it doesn't make any
# diff if we include these two as seperate, their target bbox will always be [0,0,0,0] 
# with target label as -1 :(
DATA_ROOT = os.path.join(HOME, "data/")
print("DATA_ROOT: ", DATA_ROOT)


def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on
                                 0 dim
    """
    targets = []
    imgs = []
    for sample in batch:
        imgs.append(sample[0])
        targets.append(torch.FloatTensor(sample[1]))
    return torch.stack(imgs, 0), targets


class SSDDataset(data.Dataset):
    def __init__(self, transform, phase="train"):
        """
        Args:
          phase: (boolean) train or val.
          transform: ([transforms]) image transforms.
          input_size: (int) model input size.
        """
        self.root = os.path.join(DATA_ROOT, "stage_1_train_images/")
        self.name = "RSNADataset"
        self.phase = phase
        self.transform = transform
        self.fnames = np.load(os.path.join(DATA_ROOT, 'npy_data', phase + '_fnames.npy'))
        self.boxes = np.load(os.path.join(DATA_ROOT, 'npy_data', phase + '_boxes.npy'))
        self.labels = np.load(os.path.join(DATA_ROOT, 'npy_data', phase + '_labels.npy'))
        self.num_samples = len(self.fnames)

    def __getitem__(self, idx):
        """Load image.

        Args:
          idx: (int) image index.

        Returns:
          img: (tensor) image tensor.

        """
        # Load image and boxes.
        fname = self.fnames[idx]
        boxes = np.array(self.boxes[idx]).reshape(-1, 4)
        labels = np.array(self.labels[idx]).reshape(-1, 1)
        dcm_data = pydicom.read_file(self.root + fname + ".dcm")
        img = dcm_data.pixel_array
        if self.transform is not None:
            img, boxes, labels = self.transform(img, boxes, labels)
            # pdb.set_trace()
            img = np.expand_dims(img, 0).repeat(3, axis=0)
            # img = img[:, :, (2, 1, 0)]  # to rgb
        target = np.hstack((boxes, labels))
        return torch.from_numpy(img), target

    def __len__(self):
        return self.num_samples


def provider(phase="train", batch_size=8, num_workers=4):
    dataset = SSDDataset(
        phase=phase, transform=SSDAugmentation(cfg["min_dim"], MEANS)
    )
    data_loader = data.DataLoader(
        dataset,
        batch_size,
        num_workers=num_workers,
        shuffle=True,
        collate_fn=detection_collate,
        pin_memory=True,
    )

    return data_loader


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = "None"
    dataloader = provider(phase='val', num_workers=0)
    total_iters = dataloader.__len__()
    for iter_, batch in enumerate(dataloader):
        images, boxes = batch
        print("%d/%d" % (iter_, total_iters), images.size(), len(boxes[0]))
        print(boxes)
