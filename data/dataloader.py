"""Load image/labels/boxes from an annotation file.

The list file is like:

    img.jpg xmin ymin xmax ymax label xmin ymin xmax ymax label ...
"""
from __future__ import print_function
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

CLASSES = ("abnormal",)
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
    def __init__(self, root, transform, phase="train"):
        """
        Args:
          phase: (boolean) train or val.
          transform: ([transforms]) image transforms.
          input_size: (int) model input size.
        """
        self.root = root
        self.name = "RSNADataset"
        self.phase = phase
        self.transform = transform
        df = pd.read_csv("data/stage_1_train_labels.csv")
        # df = df[df["Target"] == 0]
        df.reset_index(drop=True, inplace=True)
        df_groups = df.groupby(["patientId"]).groups
        fnames = list(df_groups.keys())
        random.seed(69)
        random.shuffle(fnames)
        # fnames = fnames[:500]
        train_fnames, val_fnames = train_test_split(fnames, test_size=0.1)
        self.fnames = train_fnames if phase == "train" else val_fnames
        self.boxes, self.labels = [], []

        for name in self.fnames:
            indices = df_groups[name]
            box = []
            for idx in indices:
                line = df.iloc[idx]
                x, y = line["x"], line["y"]
                box.append([x, y, (x + line["width"]), (y + line["height"])])
            if line["Target"]:
                self.labels.append(np.zeros(len(box)))
                self.boxes.append(box)
            else:
                self.labels.append(np.array([-1]))
                self.boxes.append([0., 0., 0., 0.])
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
    root = DATA_ROOT + "stage_1_train_images/"
    dataset = SSDDataset(
        root=root, phase=phase, transform=SSDAugmentation(cfg["min_dim"], MEANS)
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
    dataloader = provider(phase='val', num_workers=0)
    total_iters = dataloader.__len__()
    for iter_, batch in enumerate(dataloader):
        images, boxes = batch
        print("%d/%d" % (iter_, total_iters), images.size(), len(boxes[0]))
        print(boxes)
