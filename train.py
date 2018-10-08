import os
import pdb
import time
import logging
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn

from ssd import build_ssd
from layers.modules import MultiBoxLoss
from tensorboard_logger import configure
from data.config import cfg
from data.dataloader import provider
from utils import weights_init, map_iou
from extras import iter_log, epoch_log, logger


def forward(images, targets):
    if cuda:
        images = images.cuda()
        targets = [ann.cuda() for ann in targets]
    out = net(images)
    loss_l, loss_c = criterion(out, targets)
    return loss_l, loss_c


def Epoch(epoch, batch_size):
    logging.info("Starting epoch: %d " % epoch)
    for phase in ['train', 'val']:
        logging.info("Phase: %s " % phase)
        phase_bs = batch_size[phase]
        start = time.time()
        net.train(phase == "train")
        dataloader = dataloaders[phase]
        running_l_loss, running_c_loss, running_mAP = 0, 0, 0
        total_iters = len(dataloader)
        for iteration, batch in enumerate(dataloader):
            fnames, images, targets = batch
            loss_l, loss_c = forward(images, targets)
            if phase == "train":
                loss = loss_l + loss_c
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            running_l_loss += loss_l.item()
            running_c_loss += loss_c.item()
            # running_mAP += map_iou(boxes_true, boxes_pred, scores)
            if iteration % 10 == 0:
                iter_log(phase, epoch, iteration, total_iters, loss_l, loss_c, start)
        epoch_l_loss = running_l_loss / (phase_bs * total_iters)
        epoch_c_loss = running_c_loss / (phase_bs * total_iters)
        epoch_log(phase, epoch, epoch_l_loss, epoch_c_loss, start)
        dataloader = None
    return epoch_l_loss + epoch_c_loss


cuda = torch.cuda.is_available()
save_folder = "weights/6oct/"
basenet_path = "weights/vgg16_reducedfc.pth"
resume = False  # if True, will resume from weights/ckpt.pth
batch_size = {
    'train': 8,
    'val': 8
}
num_workers = 4
lr = 1e-3  # at 5e-4 it converges at 800 epochs
momentum = 0.9
weight_decay = 5e-4
best_loss = float("inf")
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
    best_loss = state["best_loss"]
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

dataloaders = {
    "train": provider(batch_size=batch_size["train"], num_workers=num_workers),
    "val": provider("val", batch_size=batch_size['val'], num_workers=num_workers),
}

for epoch in range(start_epoch, num_epochs):
    logger.info("Starting epoch: %d", epoch)
    val_loss = Epoch(epoch, batch_size)
    state = {
        "epoch": epoch,
        "best_loss": best_loss,
        "state_dict": net.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    if best_loss > val_loss:
        logging.info("New optimal found, saving state..")
        state["best_loss"] = best_loss = val_loss
        torch.save(state, os.path.join(save_folder, "model.pth"))
    torch.save(state, os.path.join(save_folder, "ckpt.pth"))

    if epoch % 10 == 0:
        state = torch.load(os.path.join(save_folder, "model.pth"))
        torch.save(state, os.path.join(save_folder, "model%d.pth" % epoch))

    logger.info("=" * 50 + "\n")

# batch_size 2 = 1.2MB, 8 = 2.2, 16 = 4.0MB
"""


"""
