import os
import pdb
import traceback
from tqdm import tqdm
import random
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from data.config import HOME


def save_npy(fnames, phase):
    print('Saving npy file for %s ...' % phase)
    labels, boxes = [], []
    for name in tqdm(fnames):
        indices = df_groups[name]
        box, label = [], []
        for idx in indices:
            line = df.iloc[idx]
            x, y = line["x"], line["y"]
            box.append([x, y, (x + line["width"]), (y + line["height"])])
            label.append(class_to_idx[class_df.iloc[idx]["class"]])
        if line["Target"]:
            labels.append(label)
            boxes.append(box)
        else:
            labels.append([-1])
            boxes.append([0., 0., 0., 0.])
    np.save(os.path.join(NPY_ROOT, phase + '_fnames.npy'), fnames)
    np.save(os.path.join(NPY_ROOT, phase + '_boxes.npy'), boxes)
    np.save(os.path.join(NPY_ROOT, phase + '_labels.npy'), labels)
    print("Done! \n")


def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)


if __name__ == '__main__':
    DATA_ROOT = os.path.join(HOME, "data")
    NPY_ROOT = os.path.join(DATA_ROOT, "npy_data")
    mkdir(NPY_ROOT)
    CLASSES = ("Lung Opacity",)  # COMMA IS FUCKING IMPORTANT
    class_to_idx = {
        "Normal": -1,
        "Lung Opacity": 0,
        "No Lung Opacity / Not Normal": 1
    }
    df = pd.read_csv(os.path.join(DATA_ROOT, "stage_1_train_labels.csv"))
    class_df = pd.read_csv(os.path.join(DATA_ROOT, 'stage_1_detailed_class_info.csv'))
    df_groups = df.groupby(["patientId"]).groups
    fnames = list(df_groups.keys())
    random.seed(69)
    random.shuffle(fnames)
    train_fnames, val_fnames = train_test_split(
        fnames, test_size=0.1, random_state=69)  # random_state is FUCKING IMPORTANT
    save_npy(val_fnames, "val")
    save_npy(train_fnames, "train")


'''
random_state is seed to train_test_split, imp to validate model perf.
'''
