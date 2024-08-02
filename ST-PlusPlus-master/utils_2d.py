import time
import sys
from pathlib import Path
import logging
import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
from tensorboardX import SummaryWriter


def count_params(model):
    param_num = sum(p.numel() for p in model.parameters())
    return param_num / 1e6


class meanIOU:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.hist = np.zeros((num_classes, num_classes))

    #计算混淆矩阵
    def _fast_hist(self, label_pred, label_true):
        mask = (label_true >= 0) & (label_true < self.num_classes)
        ##每个值出现的次数
        hist = np.bincount(
            self.num_classes * label_true[mask].astype(int) +
            label_pred[mask], minlength=self.num_classes ** 2).reshape(self.num_classes, self.num_classes)
        return hist

    def add_batch(self, predictions, gts):
        for lp, lt in zip(predictions, gts):
            self.hist += self._fast_hist(lp.flatten(), lt.flatten())

    def evaluate(self):
        ##diag函数取方阵的对角线元素，计算IOU
        iu = np.diag(self.hist) / (self.hist.sum(axis=1) + self.hist.sum(axis=0) - np.diag(self.hist))
        return iu, np.nanmean(iu)


def dice_coef(output, target):#output为预测结果 target为真实结果
    intersection = torch.sum(output * target)
    union = torch.sum(output) + torch.sum(target)
    dice = (2.0 * intersection) / (union + intersection)

    return dice.mean()


def jaccard_index(input, target):

    intersection = (input * target).long().sum()
    union = (
        input.long().sum()
        + target.long().sum()
        - intersection
    )

    if union == 0:
        return float("nan")
    else:
        return float(intersection) / float(max(union, 1))


def config_log(save_path, tensorboard=False):
    writer = SummaryWriter(str(save_path), filename_suffix=time.strftime('_%Y-%m-%d_%H-%M-%S')) if tensorboard else None

    save_path = str(Path(save_path) / 'log.txt')
    formatter = logging.Formatter('%(levelname)s [%(asctime)s] %(message)s')

    logger = logging.getLogger(save_path.split('/')[-1])
    logger.setLevel(logging.INFO)

    handler = logging.FileHandler(save_path)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    sh = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    logger.addHandler(sh)

    return logger, writer


def test_single_volume(image, label, model, classes, patch_size=[256, 256]):
    image, label = image.squeeze(0).cpu().detach(
    ).numpy(), label.squeeze(0).cpu().detach().numpy()
    prediction = np.zeros_like(label)
    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=0)
        input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
        model.eval()
        with torch.no_grad():
            output = model(input)
            if len(output) > 1:
                output = output[0]
            out = torch.argmax(torch.softmax(output, dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
            prediction[ind] = pred
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))
    return metric_list


# def test_single_volume(image, label, model, classes, patch_size=[320, 320]):
#     image, label = image.cuda(), label.squeeze().cuda()
#     model.eval()
#     with torch.no_grad():
#         output = model(image)
#         if len(output) > 1:
#             output = output[0]
#         out = torch.argmax(torch.softmax(output, dim=1), dim=1).squeeze(0)
#     dice_score = dice_coef(out, label)
#     return dice_score


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    else:
        return 0, 0
