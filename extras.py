import os
import time
import logging
from tensorboard_logger import log_value

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)

logger = logging.getLogger(__name__)


def iter_log(phase, epoch, iteration, epoch_size, loss_l, loss_c, start):
    logger.info(
        "%s epoch: %d (%d/%d) loc_loss: %.4f || cls_loss: %0.4f || %0.2f mins",
        phase,
        epoch,
        iteration,
        epoch_size,
        loss_l.item(),
        loss_c.item(),
        (time.time() - start) / 60,
    )


def epoch_log(phase, epoch, epoch_l_loss, epoch_c_loss, start):
    logger.info("%s epoch: %d finished" % (phase, epoch))
    logger.info(
        "%s Epoch: %d, loc_loss: %0.4f, cls_loss: %0.4f",
        phase,
        epoch,
        epoch_l_loss,
        epoch_c_loss,
    )
    logger.info("Time taken: %0.2f mins \n", (time.time() - start) / 60)
    log_value(phase + " cls loss", epoch_c_loss, epoch)
    log_value(phase + " loc loss", epoch_l_loss, epoch)
    log_value(phase + " total loss", epoch_c_loss + epoch_l_loss, epoch)
