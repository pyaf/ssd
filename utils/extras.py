import os
import pdb
import time
import logging
import traceback
import numpy as np
from config import HOME
from tensorboard_logger import log_value

logging.basicConfig(
    filename=os.path.join(HOME, 'weights/log.txt'),
    filemode='a',
    level=logging.DEBUG,
    format="%(asctime)s %(message)s",
    datefmt="%H:%M:%S",
)
console = logging.StreamHandler()
logger = logging.getLogger(__name__)
logger.addHandler(console)


def get_list(box):
    try:
        return [x.item() * 300 for x in box]
        return box
    except:
        traceback.print_exc()
        pdb.set_trace()


def get_pred_boxes(detections, CLS_THRESH=0.3):
    try:
        pred_boxes = []
        scores = []
        for i in np.where(detections[1, :, 0] >= CLS_THRESH)[0]:
            pred_boxes.append(get_list(detections[1, i, 1:]))
            scores.append(detections[1, i, 0].item())
        return {
            'boxes': pred_boxes,
            'scores': scores
        }
    except:
        traceback.print_exc()
        pdb.set_trace()


def print_time(start, string):
    diff = time.time() - start
    logger.info(string + ': %02d:%02d' % (diff // 60, diff % 60))

def get_gt_boxes(targets):
    try:
        gt_boxes = []
        if targets[0, -1] != -1:
            for i in range(len(targets)):
                gt_boxes.append(get_list(targets[i, :-1]))
        return gt_boxes
    except:
        traceback.print_exc()
        pdb.set_trace()


def iter_log(phase, epoch, iteration, epoch_size, loss_l, loss_c, start):
    diff = (time.time() - start)
    logger.info(
        "%s epoch: %d (%d/%d) loc_loss: %.4f || cls_loss: %0.4f || %02d:%02d",
        phase,
        epoch,
        iteration,
        epoch_size,
        loss_l.item(),
        loss_c.item(),
        diff // 60,
        diff % 60
    )


def epoch_log(phase, epoch, epoch_l_loss, epoch_c_loss, start):
    diff = (time.time() - start)
    logger.info("%s epoch: %d finished" % (phase, epoch))
    logger.info(
        "%s Epoch: %d, loc_loss: %0.4f, cls_loss: %0.4f",
        phase,
        epoch,
        epoch_l_loss,
        epoch_c_loss,
    )
    logger.info("Time taken for %s phase: %02d:%02d \n", phase, diff // 60, diff % 60)
    log_value(phase + " cls loss", epoch_c_loss, epoch)
    log_value(phase + " loc loss", epoch_l_loss, epoch)
    log_value(phase + " total loss", epoch_c_loss + epoch_l_loss, epoch)