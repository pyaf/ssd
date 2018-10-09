# -*- coding: utf-8 -*-
import pdb
import traceback
import torch
import torch.nn as nn
import torch.nn.functional as F
from data.config import cfg
from ..box_utils import match, log_sum_exp


class MultiBoxLoss(nn.Module):
    """SSD Weighted Loss Function
    Compute Targets:
        1) Produce Confidence Target Indices by matching  ground truth boxes
           with (default) 'priorboxes' that have jaccard index > threshold parameter
           (default threshold: 0.5).
        2) Produce localization target by 'encoding' variance into offsets of ground
           truth boxes and their matched  'priorboxes'.
        3) Hard negative mining to filter the excessive number of negative examples
           that comes with using a large number of default bounding boxes.
           (default negative:positive ratio 3:1)
    Objective Loss:
        L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        Where, Lconf is the CrossEntropy Loss and Lloc is the SmoothL1 Loss
        weighted by α which is set to 1 by cross val.
        Args:
            c: class confidences,
            l: predicted boxes,
            g: ground truth boxes
            N: number of matched default boxes
        See: https://arxiv.org/pdf/1512.02325.pdf for more details.
    """

    def __init__(
        self,
        num_classes,
        overlap_thresh,
        neg_pos,
        use_gpu=True,
        prior_for_matching=True,
        bkg_label=0,
        neg_mining=True,
        neg_overlap=0.5,
        encode_target=False,
    ):
        super(MultiBoxLoss, self).__init__()
        self.num_classes = num_classes
        self.threshold = overlap_thresh
        self.use_prior_for_matching = prior_for_matching  # not used
        self.background_label = bkg_label  # not used
        self.do_neg_mining = neg_mining  # not used
        self.negpos_ratio = neg_pos
        self.neg_overlap = neg_overlap  # not used
        self.encode_target = encode_target  # not used
        self.use_gpu = use_gpu
        self.variance = cfg["variance"]

    def forward(self, predictions, targets):
        """Multibox Loss
        Args:
            predictions (tuple): A tuple containing loc preds, conf preds,
            and prior boxes from SSD net.
                conf shape: torch.size(batch_size, num_priors, num_classes)
                loc shape: torch.size(batch_size, num_priors, 4)
                priors shape: torch.size(num_priors, 4)

            targets (tensor): Ground truth boxes and labels for a batch,
                shape: [batch_size, num_objs, 5] (last idx is the label).
        """
        try:
            # pdb.set_trace()
            loc_data, conf_data, priors = predictions
            batch = loc_data.size(0)  # batch size
            priors = priors[: loc_data.size(1), :]  # useless step ?
            num_priors = priors.size(0)  # 8732

            # match priors (default boxes) and ground truth boxes
            loc_t = torch.Tensor(batch, num_priors, 4)
            conf_t = torch.LongTensor(batch, num_priors)
            for idx in range(batch):
                truths = targets[idx][:, :-1].data
                labels = targets[idx][:, -1].data
                defaults = priors.data
                loc_t[idx], conf_t[idx] = match(
                    self.threshold,
                    truths,
                    defaults,
                    self.variance,
                    labels,
                    loc_t,
                    conf_t,
                )
            if self.use_gpu:
                loc_t = loc_t.cuda()
                conf_t = conf_t.cuda()
            # wrap targets (not required in PT 0.4)
            # loc_t = Variable(loc_t, requires_grad=False)
            # conf_t = Variable(conf_t, requires_grad=False)

            pos = conf_t > 0  # [batch, num_priors]
            num_pos = pos.long().sum(dim=1, keepdim=True)  # [batch, 1]
            # num_pos =  num of +ve priors for each image

            # ************ loc loss **********
            # Localization Loss (Smooth L1), calculated only for +ve priors
            pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
            # ^ [batch, num_priors, 4]; `pos` copied 4 times
            loc_p = loc_data[pos_idx].view(-1, 4)  # [num_pos, 4]
            loc_t = loc_t[pos_idx].view(-1, 4)  # [num_pos, 4]
            loss_l = F.smooth_l1_loss(loc_p, loc_t, reduction="sum")

            # ********** Conf loss ***********
            # Compute max conf across batch for hard negative mining
            batch_conf = conf_data.view(-1, self.num_classes)
            # ^ [batch * num_priors, num_classes]
            loss_c = log_sum_exp(batch_conf) - \
                batch_conf.gather(1, conf_t.view(-1, 1))
            # gather values in batch_conf acc to values in conf_t as index
            # loss_c.shape => [batch * num_priors, 1]

            # Hard Negative Mining
            loss_c[pos.view(-1, 1)] = 0  # filter out pos boxes for now
            loss_c = loss_c.view(batch, -1)  # [batch, num_priors]
            _, loss_idx = loss_c.sort(dim=1, descending=True)  # high to low
            # ^ [batch, num_priors]
            _, idx_rank = loss_idx.sort(dim=1)  # [batch, num_priors]
            # idx rank in org loss_c
            # num_pos = pos.long().sum(dim=1, keepdim=True)  # [batch, 1]
            num_neg = torch.clamp(
                self.negpos_ratio * num_pos, max=pos.size(1) - 1, min=10)
            # num_neg.shape => [batch, 1]
            # num_neg = 3 * num_pos; pos.size(1) = num_priors
            neg = idx_rank < num_neg.expand_as(idx_rank)
            # num_neg.expand_as(idx_rank) will broadcast num_neg to idx_rank's shape

            # Confidence Loss Including Positive and Negative Examples
            pos_idx = pos.unsqueeze(2).expand_as(conf_data)
            neg_idx = neg.unsqueeze(2).expand_as(conf_data)
            # pos_idx, neg_idx => [batch, num_priors, num_classes]
            conf_p = conf_data[(pos_idx + neg_idx).gt(0)].view(-1, self.num_classes)
            targets_weighted = conf_t[(pos + neg).gt(0)]
            loss_c = F.cross_entropy(conf_p, targets_weighted, reduction="sum")

            # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N

            # import pdb; pdb.set_trace()
            N_p = num_pos.sum().item()
            N_n = num_neg.sum().item()
            loss_l /= N_p if N_p else 1  # avoid 0/0
            loss_c /= N_p + N_n
            # if loss_c == 0:
            #     pdb.set_trace()
            return loss_l, loss_c
        except Exception as e:
            print(e)
            traceback.print_exc()
            pdb.set_trace()


'''
gather in pytorch: https://stackoverflow.com/a/51032153/6300703
'''
