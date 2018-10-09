import matplotlib; matplotlib.use('Agg')
import os
import pdb
import time
import _thread
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
from shutil import copyfile


class Model(object):
    def __init__(self):
        self.resume = False
        folder = '9oct'
        self.num_workers = 4
        self.batch_size = {'train': 8, 'val': 8}
        self.lr = 1e-3
        self.momentum = 0.9
        self.weight_decay = 5e-4
        self.best_loss = float("inf")
        self.best_mAP = 0
        self.start_epoch = 0
        self.num_epochs = 1000
        self.IOU_THRESH = 0.3  # for +ve/-ve anchor boxes
        self.CLS_THRESH = 0.15  # used in detection
        self.NMS_THRESH = 0.45  # IoU threshold in NMS
        self.NEG_POS_RATIO = 3  # for HNM
        self.top_k = 10  # top k bboxes to be detected
        self.phases = ['train', 'val']
        self.cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.cuda else "cpu")
        weights_path = os.path.join(os.getcwd(), "weights")
        self.save_folder = os.path.join(weights_path, folder)
        self.basenet_path = os.path.join(weights_path, "vgg16_reducedfc.pth")
        self.model_path = os.path.join(self.save_folder, "model.pth")
        self.tensor_type = "torch.cuda.FloatTensor" if self.cuda else "torch.FloatTensor"
        torch.set_default_tensor_type(self.tensor_type)
        self.softmax = nn.Softmax(dim=-1)
        self.detect = Detect(cfg['num_classes'], self.top_k, self.CLS_THRESH, self.NMS_THRESH)
        self.net = build_ssd("train", cfg["min_dim"], cfg["num_classes"])
        self.optimizer = optim.SGD(self.net.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)
        self.criterion = MultiBoxLoss(cfg["num_classes"], self.IOU_THRESH, self.NEG_POS_RATIO, self.cuda)
        self.log = logger.info
        if self.resume:
            self.resume_net()
        else:
            self.initialize_net()
        self.net = self.net.to(self.device)
        if self.cuda: 
            cudnn.benchmark = True
        configure(os.path.join(self.save_folder, "logs"), flush_secs=5)
        self.reset()

    def reset(self):
        self.predicted_boxes = defaultdict(dict)
        self.ground_truth_boxes = defaultdict(list)

    def resume_net(self):
        self.resume_path = os.path.join(self.save_folder, "ckpt.pth")
        self.log("Resuming training, loading {} ...".format(self.resume_path))
        state = torch.load(self.resume_path)
        self.net.load_state_dict(state["state_dict"])
        self.optimizer.load_state_dict(state["optimizer"])
        # best_loss = state["best_loss"]
        self.best_mAP = state["best_mAP"]
        self.start_epoch = state["epoch"] + 1

    def initialize_net(self):
        vgg_weights = torch.load(self.basenet_path)
        self.log("Loading base network...")
        self.net.vgg.load_state_dict(vgg_weights)
        self.log("Initializing weights...")
        self.net.extras.apply(weights_init)
        self.net.loc.apply(weights_init)
        self.net.conf.apply(weights_init)

    def forward(self, images, targets):
        images = images.to(self.device)
        targets = [ann.to(self.device) for ann in targets]
        outputs = self.net(images)
        loss_l, loss_c = self.criterion(outputs, targets)
        return loss_l, loss_c, outputs

    def iterate(self, epoch, phase):
        self.reset()
        self.log("Starting epoch: %d | phase: %s " % (epoch, phase))
        batch_size = self.batch_size[phase]
        start = time.time()
        self.net.train(phase == "train")
        dataloader = provider(phase, batch_size=batch_size, num_workers=self.num_workers)
        running_l_loss, running_c_loss = 0, 0
        total_iters = len(dataloader)
        total_images = dataloader.dataset.num_samples
        for iteration, batch in enumerate(dataloader):
            fnames, images, targets = batch
            loss_l, loss_c, outputs = self.forward(images, targets)
            if phase == "train":
                self.optimizer.zero_grad()
                loss = loss_c + loss_l
                loss.backward()
                self.optimizer.step()
            running_l_loss += loss_l.item()
            running_c_loss += loss_c.item()
            _thread.start_new_thread(self.update_boxes, (fnames, outputs, targets))
            if iteration % 10 == 0:
                iter_log(phase, epoch, iteration, total_iters, loss_l, loss_c, start)
        epoch_l_loss = running_l_loss / total_images
        epoch_c_loss = running_c_loss / total_images
        while len(self.ground_truth_boxes) != total_images:  # thread locha
            self.log('Waiting for threads to get over')
            # pdb.set_trace()
            time.sleep(1)
        mAP = get_mAP(self.ground_truth_boxes, self.predicted_boxes)
        epoch_log(phase, mAP, epoch, epoch_l_loss, epoch_c_loss, start)
        del dataloader
        return (epoch_l_loss + epoch_c_loss, mAP)

    def update_boxes(self, fnames, outputs, targets):
        outputs[1] = self.softmax(outputs[1])
        detections = self.detect(* outputs).data
        for i, name in enumerate(fnames):  # len(images) no batch_size (last iter issue)
            self.predicted_boxes[name] = get_pred_boxes(detections[i])
            self.ground_truth_boxes[name] = get_gt_boxes(targets[i])

    def train(self):
        for epoch in range(self.start_epoch, self.num_epochs):
            self.log("Starting epoch: %d", epoch)
            self.iterate(epoch, 'train')
            val_loss, mAP = self.iterate(epoch, 'val')
            state = {
                "epoch": epoch,
                "best_mAP": self.best_mAP,
                "state_dict": self.net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            }
            if mAP > self.best_mAP:
                self.log("New optimal found, saving state..")
                state["best_mAP"] = self.best_mAP = mAP
                torch.save(state, os.path.join(self.save_folder, "model.pth"))
            torch.save(state, os.path.join(self.save_folder, "ckpt.pth"))
            if epoch and epoch % 10 == 0:
                copyfile(self.model_path, os.path.join(self.save_folder, "model%d.pth" % epoch))
            self.log("=" * 50 + "\n")


if __name__ == '__main__':
    model = Model()
    model.train()

# batch_size 2 = 1.2MB, 8 = 2.2, 16 = 4.0MB
"""


"""
