import matplotlib; matplotlib.use('Agg')
import os
import pdb
import time
import logging
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from collections import defaultdict
from ssd import build_ssd
import torch.nn as nn
from layers import Detect
from layers.modules import MultiBoxLoss
from tensorboard_logger import configure
from data.config import cfg
from data.dataloader import provider
from utils import weights_init
from extras import iter_log, epoch_log, logger, get_pred_boxes, get_gt_boxes
from metrics import get_mAP


def forward(images, targets):
    if cuda:
        images = images.cuda()
        targets = [ann.cuda() for ann in targets]
    out = net(images)
    loss_l, loss_c = criterion(out, targets)
    out[1] = softmax(out[1])
    detections = detect(*out).data
    return loss_l, loss_c, detections


def Epoch(epoch, batch_size):
    logging.info("Starting epoch: %d " % epoch)
    for phase in ['train', 'val']:
        logging.info("Phase: %s " % phase)
        phase_bs = batch_size[phase]
        start = time.time()
        net.train(phase == "train")
        dataloader = provider(phase, batch_size=phase_bs, num_workers=num_workers)
        running_l_loss, running_c_loss = 0, 0
        total_iters = len(dataloader)
        predicted_boxes = defaultdict(dict)
        ground_truth_boxes = defaultdict(list)
        for iteration, batch in enumerate(dataloader):
            fnames, images, targets = batch
            loss_l, loss_c, detections = forward(images, targets)
            if phase == "train":
                loss = loss_l + loss_c
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            running_l_loss += loss_l.item()
            running_c_loss += loss_c.item()
            for i, name in enumerate(fnames):  # len(images) no batch_size (last iter issue)
                predicted_boxes[name] = get_pred_boxes(detections[i])
                ground_truth_boxes[name] = get_gt_boxes(targets[i])
            if iteration % 10 == 0:
                iter_log(phase, epoch, iteration, total_iters, loss_l, loss_c, start)
        epoch_l_loss = running_l_loss / (phase_bs * total_iters)
        epoch_c_loss = running_c_loss / (phase_bs * total_iters)
        print('calculating mAP...')
        mAP = get_mAP(ground_truth_boxes, predicted_boxes)
        epoch_log(phase, mAP, epoch, epoch_l_loss, epoch_c_loss, start)
        del dataloader
    return (epoch_l_loss + epoch_c_loss, mAP)


cuda = torch.cuda.is_available()
save_folder = "weights/9oct/"
basenet_path = "weights/vgg16_reducedfc.pth"
resume = False  # if True, will resume from weights/ckpt.pth
batch_size = {
    'train': 2,
    'val': 2
}
num_workers = 4
lr = 1e-3  # at 5e-4 it converges at 800 epochs
momentum = 0.9
weight_decay = 5e-4
best_loss = float("inf")
best_mAP = 0
start_epoch = 0
num_epochs = 1000
pos_prior_threshold = 0.3

testing = False
if testing:
    batch_size = {
        'train': 2,
        'val': 2
    }
    num_workers = 0

configure(os.path.join(save_folder, "logs"), flush_secs=5)
tensor_type = "torch.cuda.FloatTensor" if cuda else "torch.FloatTensor"
torch.set_default_tensor_type(tensor_type)
softmax = nn.Softmax(dim=-1)
detect = Detect(cfg['num_classes'], 0, 200, 0.01, 0.45)
net = build_ssd("train", cfg["min_dim"], cfg["num_classes"])
optimizer = optim.SGD(
    net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
)
criterion = MultiBoxLoss(
    cfg["num_classes"], pos_prior_threshold, True, 0, True, 3, 0.5, False, cuda
)

if resume:
    resume_path = os.path.join(save_folder, "ckpt.pth")
    logger.info("Resuming training, loading {} ...".format(resume_path))
    state = torch.load(resume_path)
    net.load_state_dict(state["state_dict"])
    optimizer.load_state_dict(state["optimizer"])
    # best_loss = state["best_loss"]
    best_mAP = state["best_mAP"]
    start_epoch = state["epoch"] + 1
else:
    vgg_weights = torch.load(basenet_path)
    logger.info("Loading base network...")
    net.vgg.load_state_dict(vgg_weights)
    logger.info("Initializing weights...")
    net.extras.apply(weights_init)
    net.loc.apply(weights_init)
    net.conf.apply(weights_init)

if cuda:
    net = net.cuda()
    cudnn.benchmark = True


for epoch in range(start_epoch, num_epochs):
    logger.info("Starting epoch: %d", epoch)
    val_loss, mAP = Epoch(epoch, batch_size)
    state = {
        "epoch": epoch,
        # "best_loss": best_loss,
        "best_mAP": best_mAP,
        "state_dict": net.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    # if best_loss > val_loss:
    if mAP > best_mAP:
        logging.info("New optimal found, saving state..")
        # state["best_loss"] = best_loss = val_loss
        state["best_mAP"] = best_mAP = mAP
        torch.save(state, os.path.join(save_folder, "model.pth"))
    torch.save(state, os.path.join(save_folder, "ckpt.pth"))

    if epoch and epoch % 10 == 0:
        state = torch.load(os.path.join(save_folder, "model.pth"))
        torch.save(state, os.path.join(save_folder, "model%d.pth" % epoch))

    logger.info("=" * 50 + "\n")

# batch_size 2 = 1.2MB, 8 = 2.2, 16 = 4.0MB
"""


"""
