import base64
import math
import random
from pathlib import Path

from medpy import metric
from torchvision import transforms
from data.dataset1 import BaseDataSets, RandomGenerator, TwoStreamBatchSampler, BaseSTDataSets
from model.unet1 import UNet, UNet_UMMC
from utils_2d import count_params, meanIOU, config_log, test_single_volume, dice_coef
import ramps

import argparse
from copy import deepcopy
import numpy as np
import torch.backends.cudnn as cudnn
import torch
from PIL import Image
from torch import nn, optim
from loss import DiceLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
import os


"""Global Variables"""
result_dir = 'result/'
pretraining_epochs, semitraining_epochs, finaltraining_epochs = 20, 100, 100
pred_step, save_step = 10, 5
ce_loss = nn.CrossEntropyLoss()
CE_loss_r = nn.CrossEntropyLoss(reduction='none')
dice_loss = DiceLoss(n_classes=4)
kl_distance = nn.KLDivLoss(reduction='none')
logger = None


def parse_args():
    parser = argparse.ArgumentParser(description='ST and ST++ Framework')

    # basic settings
    parser.add_argument('--root-path', type=str, default=r'D:\shujuji\ACDC')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.003)
    parser.add_argument('--max_iterations', type=int,
                        default=30000, help='maximum epoch number to train')
    parser.add_argument('--patch_size', type=list, default=[256, 256], help='patch size of network input')
    # parser.add_argument('--seed', type=int, default=1336, help='random seed')
    parser.add_argument('--num_classes', type=int, default=4, help='output channel of network')
    parser.add_argument('--deterministic', type=int, default=1,
                        help='whether use deterministic training')
    parser.add_argument('--gpu', type=str, default='0', help='GPU to use')

    # semi-supervised settings
    parser.add_argument('--labelnum', type=int, default=14, help='labeled data')
    parser.add_argument('--reliable-id-path', type=str, default=r'D:\shujuji\ACDC')

    # costs
    parser.add_argument('--consistency', type=float,
                        default=0.1, help='consistency')
    parser.add_argument('--consistency_rampup', type=float,
                        default=40.0, help='consistency_rampup')
    parser.add_argument('--ema_decay1', type=float, default=0.999, help='ema_decay')
    parser.add_argument('--ema_decay2', type=float, default=0.99, help='ema_decay')

    args = parser.parse_args()
    return args


def patients_to_slices(dataset, patiens_num):
    ref_dict = None
    if "ACDC" in dataset:
        ref_dict = {"3": 68, "7": 136,
                    "14": 256, "21": 396, "28": 512, "35": 664, "70": 1312}
    elif "isic":
        ref_dict = {"20": 377, "40": 754, "100": 1886}
    else:
        print("Error")
    return ref_dict[str(patiens_num)]


def get_current_consistency_weight(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))


@torch.no_grad()
def update_ema_variables(model, ema_model, alpha):
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_((1 - alpha) * param.data)


def main(args, model_path1, model_path2, model_path3):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    labeled_slice = patients_to_slices(args.root_path, args.labelnum)
    db_train = BaseDataSets(base_dir=args.root_path,
                            split="train",
                            num=labeled_slice,
                            transform=transforms.Compose([
                                RandomGenerator(args.patch_size)
                            ]))
    db_val = BaseDataSets(base_dir=args.root_path, split="val")
    trained_slice = len(db_train)
    print("pretrained slices is: {}".format(trained_slice))
    trainloader = DataLoader(db_train, batch_size=4, num_workers=0, pin_memory=True)
    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=0)
    model = UNet_UMMC(in_chns=1, class_num=args.num_classes).cuda()
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=0.9, weight_decay=0.0001)
    print('\nParams: %.1fM' % count_params(model))

    ##最初始的教师模型
    best_model, checkpoints = pretrain(model, optimizer, trainloader, valloader, db_val, args, model_path1)

    # <===================================== Select Reliable IDs =====================================>
    print('\n\n\n================> Total stage 2/4: Select reliable images for the 1st stage re-training')
    db_label = BaseDataSets(base_dir=args.root_path, split="label", num=labeled_slice,
                            transform=transforms.Compose([
                                RandomGenerator(args.patch_size)
                            ]))
    unlabeled_slice = len(db_label)
    print("unlabeled slices is: {}".format(unlabeled_slice))
    labelloader = DataLoader(db_label, batch_size=1, shuffle=False, num_workers=0)

    select_reliable(checkpoints, labelloader, args)

    print('\n\n\n================> Total stage 3/4: training on labeled and reliable unlabeled images')
    semilabel_slice = math.ceil(unlabeled_slice / 2)
    db_train = BaseDataSets(base_dir=args.root_path,
                            split="train",
                            num=labeled_slice,
                            transform=transforms.Compose([
                                RandomGenerator(args.patch_size)
                            ]))
    db_semitrain = BaseDataSets(base_dir=args.reliable_id_path,
                                split="semitrain",
                                num=semilabel_slice,
                                transform=transforms.Compose([
                                    RandomGenerator(args.patch_size)
                                ]))
    semilabel_slice = len(db_semitrain)
    print("labeled silices is: {}, unlabeled slices is: {}".format(
        labeled_slice, semilabel_slice))
    db_train.sample_list = db_train.sample_list * math.ceil(len(db_semitrain) / len(db_train))
    trainloader = DataLoader(db_train, batch_size=4, num_workers=0, pin_memory=True)
    semitrainloader = DataLoader(db_semitrain, batch_size=1, num_workers=0, pin_memory=True)
    new_loader1, Dice1 = pred_unlabel(best_model, semitrainloader, args.batch_size)
    logger.info('Pseudo label dice : {}'.format(Dice1))
    semi_model = UNet_UMMC(in_chns=1, class_num=args.num_classes).cuda()
    semi_optimizer = optim.SGD(semi_model.parameters(), lr=args.lr,
                          momentum=0.9, weight_decay=0.0001)
    best_model = semi_train(semi_model, semi_optimizer, trainloader, semitrainloader, new_loader1, valloader, db_val, args, model_path2)

    print('\n\n\n================> Total stage 4/4: training on labeled and All unlabeled images')
    db_train = BaseDataSets(base_dir=args.root_path,
                            split="train",
                            num=labeled_slice,
                            transform=transforms.Compose([
                                RandomGenerator(args.patch_size)
                            ]))
    db_finaltrain = BaseDataSets(base_dir=args.reliable_id_path,
                                 split="finaltrain",
                                 num=None,
                                 transform=transforms.Compose([
                                     RandomGenerator(args.patch_size)
                                 ]))
    finallabel_slice = len(db_finaltrain)
    print("labeled silices is: {}, unlabeled slices is: {}".format(
        labeled_slice, finallabel_slice))
    db_train.sample_list = db_train.sample_list * math.ceil(len(db_finaltrain) / len(db_train))
    trainloader = DataLoader(db_train, batch_size=4, num_workers=0, pin_memory=True)
    finaltrainloader = DataLoader(db_finaltrain, batch_size=1, num_workers=0, pin_memory=True)
    new_loader1, Dice1 = pred_unlabel(best_model, finaltrainloader, args.batch_size)
    logger.info('Pseudo label dice : {}'.format(Dice1))
    final_model = UNet_UMMC(in_chns=1, class_num=args.num_classes).cuda()
    final_optimizer = optim.SGD(final_model.parameters(), lr=args.lr,
                               momentum=0.9, weight_decay=0.0001)
    best_model = final_train(final_model, final_optimizer, trainloader, finaltrainloader, new_loader1, valloader, db_val, args, model_path3)


def pretrain(model, optimizer, trainloader, valloader, db_val, args, model_path1):
    iters = 0
    total_iters = len(trainloader) * pretraining_epochs

    'model_path'
    model_path1 = Path(model_path1)
    model_path1.mkdir(exist_ok=True)

    """Create Path"""
    save_path = Path(result_dir) / 'pretrain'
    save_path.mkdir(exist_ok=True)

    """Create logger"""
    global logger
    logger, writer = config_log(save_path, tensorboard=True)
    logger.info("Pretrain, save path : " + str(save_path))

    """Training"""
    checkpoints = []
    best_performance = 0.0
    model.train()
    for epoch in tqdm(range(1, pretraining_epochs + 1), ncols=70):
        if epoch % save_step == 0:
            """Testing"""
            model.eval()
            metric_list = 0.0
            for i_batch, sampled_batch in enumerate(valloader):
                metric_i = test_single_volume(sampled_batch['image'], sampled_batch['label'], model,
                                              classes=args.num_classes)
                metric_list += np.array(metric_i)
            metric_list = metric_list / len(db_val)

            performance = np.mean(metric_list, axis=0)[0]
            mean_hd95 = np.mean(metric_list, axis=0)[1]
            writer.add_scalar('info/val_mean_dice', performance, epoch)
            writer.add_scalar('info/val_mean_hd95', mean_hd95, epoch)
            if performance > best_performance:
                best_performance = performance
                save_mode_path = os.path.join(model_path1,
                                              'epoch_{}_dice_{}.pth'.format(
                                                  epoch, round(best_performance, 4)))
                save_best = os.path.join(model_path1, 'best_model.pth')
                torch.save(model.state_dict(), save_mode_path)
                torch.save(model.state_dict(), save_best)
                best_model = deepcopy(model)
            logger.info('epoch %d : mean_dice : %f mean_hd95 : %f' % (epoch, performance, mean_hd95))
            model.train()

        for i_batch, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

            outputs, outputs_aux1 = model(volume_batch)

            outputs_soft1 = torch.softmax(outputs, dim=1)
            outputs_soft2 = torch.softmax(outputs_aux1, dim=1)

            loss_ce1 = ce_loss(outputs, label_batch.long())
            loss_ce2 = ce_loss(outputs_aux1, label_batch.long())
            loss_dice1 = dice_loss(outputs_soft1, label_batch.unsqueeze(1))
            loss_dice2 = dice_loss(outputs_soft2, label_batch.unsqueeze(1))

            loss = (loss_ce1 + loss_ce2 + loss_dice1 + loss_dice2) / 4

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            iters += 1
            lr_ = args.lr * (1 - iters / total_iters) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_
            logger.info('epoch %d : loss : %03f, loss_ce1: %03f,  loss_dice1: %03f, loss_ce2: %03f, loss_dice2: %03f'
                        % (epoch, loss, loss_ce1, loss_dice1, loss_ce2, loss_dice2))

        if ((epoch + 1) in [pretraining_epochs // 3, pretraining_epochs // 2, pretraining_epochs * 2 // 3, pretraining_epochs]):
            checkpoints.append(deepcopy(model))

    writer.close()
    return best_model, checkpoints


def select_reliable(models, dataloader, args):
    if not os.path.exists(args.reliable_id_path):
        os.makedirs(args.reliable_id_path)

    for i in range(len(models)):
        models[i].eval()

    id_to_reliability = []
    Count = []
    with torch.no_grad():
        for sampled_batch, case in tqdm(dataloader):
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

            preds = []
            for model in models:
                pred = model(volume_batch)
                if len(pred) > 1:
                    pred = pred[0]
                preds.append(torch.softmax(pred, dim=1))

            Uncertainty = []
            for i in range(len(preds) - 1):
                variance = torch.mean((preds[i] - preds[-1]) ** 2)
                Uncertainty.append(variance)
            mypred = preds[-1].to('cpu')
            count = (mypred > 0.8).sum().item()
            Count.append(count)
            reliability = sum(Uncertainty) / len(Uncertainty)
            reliability = reliability.to('cpu').item()
            print(reliability)
            id_to_reliability.append((case[0], str(reliability), str(count)))
    length = len(id_to_reliability)
    id_to_reliability.sort(key=lambda elem: elem[1], reverse=False)
    new_reliability = id_to_reliability[round(length / 4):round((length * 3) / 4)]
    new_reliability.sort(key=lambda elem: elem[2], reverse=True)
    id_to_reliability[round(length / 4):round((length * 3) / 4)] = new_reliability
    with open(os.path.join(args.reliable_id_path, 'reliable.list'), 'w') as f:
        for elem in id_to_reliability:
            ##图片名称
            f.write(elem[0] + ' ' + elem[1] + ' ' + elem[2] + '\n')


@torch.no_grad()
def pred_unlabel(model, pred_loader, batch_size):
    unimgs, preds = [], []
    plab_dice = []
    model.eval()
    for step, sampled_batch in enumerate(pred_loader):
        volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
        volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

        outputs, outputs_aux1 = model(volume_batch)
        d1_out1_soft = torch.softmax(outputs, dim=1)
        pseudo_outputs1 = torch.argmax(d1_out1_soft, dim=1, keepdim=True)
        unimgs.append(volume_batch)
        preds.append(pseudo_outputs1)
        pseudo_outputs1 = pseudo_outputs1.squeeze()
        label_batch = label_batch.squeeze()
        score = torch.mean(((pseudo_outputs1 - label_batch) ** 2).float())
        plab_dice.append(score)
        print('Pseudo label dice : step_{}_dice_{}'.format(step, score))
    Dice = sum(plab_dice) / len(plab_dice)
    new_loader = DataLoader(BaseSTDataSets(unimgs, preds), batch_size=batch_size, shuffle=False, num_workers=0)
    return new_loader, Dice


def semi_train(model, optimizer, trainloader, semitrainloader, new_loader1, valloader, db_val, args, model_path2):
    iters = 0
    total_iters = len(trainloader) * semitraining_epochs

    'model_path'
    model_path2 = Path(model_path2)
    model_path2.mkdir(exist_ok=True)

    """Create Path"""
    save_path = Path(result_dir) / 'semi-info'
    save_path.mkdir(exist_ok=True)

    """Create logger"""
    global logger
    logger, writer = config_log(save_path, tensorboard=True)
    logger.info("semi_train, save path : " + str(save_path))

    """load model"""
    save_best = os.path.join(Path(model_path1), 'best_model.pth')
    model.load_state_dict(torch.load(save_best), strict=False)

    best_performance = 0.0
    model.train()
    for epoch in tqdm(range(0, semitraining_epochs), ncols=70):
        if epoch % save_step == 0 and epoch > 0:
            """Testing"""
            model.eval()
            metric_list = 0.0
            for i_batch, sampled_batch in enumerate(valloader):
                metric_i = test_single_volume(sampled_batch['image'], sampled_batch['label'], model,
                                              classes=args.num_classes)
                metric_list += np.array(metric_i)
            metric_list = metric_list / len(db_val)

            performance = np.mean(metric_list, axis=0)[0]
            mean_hd95 = np.mean(metric_list, axis=0)[1]
            writer.add_scalar('info/val_mean_dice', performance, epoch)
            writer.add_scalar('info/val_mean_hd95', mean_hd95, epoch)
            if performance > best_performance:
                best_performance = performance
                save_mode_path = os.path.join(model_path1,
                                              'epoch_{}_dice_{}.pth'.format(
                                                  epoch, round(best_performance, 4)))
                save_best = os.path.join(model_path1, 'best_model.pth')
                torch.save(model.state_dict(), save_mode_path)
                torch.save(model.state_dict(), save_best)
                best_model = deepcopy(model)
            logger.info('epoch %d : mean_dice : %f mean_hd95 : %f' % (epoch, performance, mean_hd95))
            model.train()

        """Predict pseudo labels"""
        if epoch % pred_step == 0 and epoch > 0:
            logger.info('Starting predict unlab')
            new_loader1, Dice1 = pred_unlabel(model, semitrainloader, args.batch_size)
            logger.info('Pseudo label dice : {}'.format(Dice1))

        """Training"""
        for i_batch, ((labeled_batch), (unlabeled_batch1)) in enumerate(zip(trainloader, new_loader1)):
            volume_batch, label_batch = labeled_batch['image'], labeled_batch['label']
            unvolume_batch1, pred_batch1 = unlabeled_batch1['image'], unlabeled_batch1['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            unvolume_batch1, pred_batch1 = unvolume_batch1.cuda(), pred_batch1.cuda()

            '''Supervised Loss'''
            outputs, outputs_aux1 = model(volume_batch)

            d1_out1_soft = torch.softmax(outputs, dim=1)
            d1_out2_soft = torch.softmax(outputs_aux1, dim=1)

            loss_ce1 = ce_loss(outputs, label_batch.long())
            loss_ce2 = ce_loss(outputs_aux1, label_batch.long())
            loss_dice1 = DiceLoss(d1_out1_soft, label_batch.unsqueeze(1))
            loss_dice2 = DiceLoss(d1_out2_soft, label_batch.unsqueeze(1))

            supervised_loss = (loss_ce1 + loss_ce2 + loss_dice1 + loss_dice2) / 4

            '''UnSupervised Loss'''
            un_outputs, un_outputs_aux1 = model(unvolume_batch1)
            un_out1_soft = torch.softmax(un_outputs, dim=1)
            un_out2_soft = torch.softmax(un_outputs_aux1, dim=1)

            variance = torch.sum(kl_distance(
                torch.log(un_out1_soft + 1e-8), un_out2_soft), dim=1, keepdim=True)
            exp_variance = torch.exp(-variance)
            unloss_ce = torch.mean(
                CE_loss_r(un_outputs, pred_batch1.long()) * exp_variance) + torch.mean(variance)

            pseudo_supervision_loss = unloss_ce

            loss = supervised_loss + pseudo_supervision_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            iters += 1
            lr_ = args.lr * (1 - iters / total_iters) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_
            logger.info('epoch %d : loss : %03f, supervised_loss: %03f, pseudo_supervision_loss: %03f'
                % (epoch, loss, supervised_loss, pseudo_supervision_loss))

    return best_model


def final_train(model, optimizer, trainloader, finaltrainloader, new_loader1, valloader, db_val, args, model_path3):
    iters = 0
    total_iters = len(trainloader) * finaltraining_epochs

    'model_path'
    model_path3 = Path(model_path3)
    model_path3.mkdir(exist_ok=True)

    """Create Path"""
    save_path = Path(result_dir) / 'final_train'
    save_path.mkdir(exist_ok=True)

    """Create logger"""
    global logger
    logger, writer = config_log(save_path, tensorboard=True)
    logger.info("final_train, save path : " + str(save_path))

    """load model"""
    save_best = os.path.join(Path(model_path2), 'best_model.pth')
    model.load_state_dict(torch.load(save_best))

    best_performance = 0.0
    model.train()
    for epoch in tqdm(range(0, finaltraining_epochs), ncols=70):
        if epoch % save_step == 0 and epoch > 0:
            """Testing"""
            model.eval()
            metric_list = 0.0
            for i_batch, sampled_batch in enumerate(valloader):
                metric_i = test_single_volume(sampled_batch['image'], sampled_batch['label'], model,
                                              classes=args.num_classes)
                metric_list += np.array(metric_i)
            metric_list = metric_list / len(db_val)

            performance = np.mean(metric_list, axis=0)[0]
            mean_hd95 = np.mean(metric_list, axis=0)[1]
            writer.add_scalar('info/val_mean_dice', performance, epoch)
            writer.add_scalar('info/val_mean_hd95', mean_hd95, epoch)
            if performance > best_performance:
                best_performance = performance
                save_mode_path = os.path.join(model_path1,
                                              'epoch_{}_dice_{}.pth'.format(
                                                  epoch, round(best_performance, 4)))
                save_best = os.path.join(model_path1, 'best_model.pth')
                torch.save(model.state_dict(), save_mode_path)
                torch.save(model.state_dict(), save_best)
                best_model = deepcopy(model)
            logger.info('epoch %d : mean_dice : %f mean_hd95 : %f' % (epoch, performance, mean_hd95))
            model.train()
        """Predict pseudo labels"""
        if epoch % pred_step == 0 and epoch > 0:
            logger.info('Starting predict unlab')
            new_loader1, Dice1 = pred_unlabel(model, finaltrainloader, args.batch_size)
            logger.info('Pseudo label dice : {}'.format(Dice1))

        """Training"""
        for i_batch, ((labeled_batch), (unlabeled_batch1)) in enumerate(zip(trainloader, new_loader1)):
            volume_batch, label_batch = labeled_batch['image'], labeled_batch['label']
            unvolume_batch1, pred_batch1 = unlabeled_batch1['image'], unlabeled_batch1['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            unvolume_batch1, pred_batch1 = unvolume_batch1.cuda(), pred_batch1.cuda()

            '''Supervised Loss'''
            outputs, outputs_aux1 = model(volume_batch)

            d1_out1_soft = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
            d1_out2_soft = torch.argmax(torch.softmax(outputs_aux1, dim=1), dim=1, keepdim=True)

            loss_ce1 = ce_loss(outputs, label_batch.long())
            loss_ce2 = ce_loss(outputs_aux1, label_batch.long())
            loss_dice1 = DiceLoss(d1_out1_soft, label_batch.unsqueeze(1))
            loss_dice2 = DiceLoss(d1_out2_soft, label_batch.unsqueeze(1))

            supervised_loss = (loss_ce1 + loss_ce2 + loss_dice1 + loss_dice2) / 4

            '''UnSupervised Loss'''
            un_outputs, un_outputs_aux1 = model(unvolume_batch1)
            un_out1_soft = torch.softmax(un_outputs, dim=1)
            un_out2_soft = torch.softmax(un_outputs_aux1, dim=1)

            variance = torch.sum(kl_distance(
                torch.log(un_out1_soft + 1e-8), un_out2_soft), dim=1, keepdim=True)
            exp_variance = torch.exp(-variance)
            unloss_ce = torch.mean(
                CE_loss_r(un_outputs, pred_batch1.long()) * exp_variance) + torch.mean(variance)

            pseudo_supervision_loss = unloss_ce

            loss = supervised_loss + pseudo_supervision_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            iters += 1
            lr_ = args.lr * (1 - iters / total_iters) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_
            logger.info('epoch %d : loss : %03f, supervised_loss: %03f, pseudo_supervision_loss: %03f'
                % (epoch, loss, supervised_loss, pseudo_supervision_loss))

    return best_model


if __name__ == '__main__':
    args = parse_args()
    for i in [1337, 1339, 520, 521]:
        fix_seed = i
        model_path1 = './pretrained/'+str(fix_seed)
        model_path2 = './semitrained/'+str(fix_seed)
        model_path3 = './finaltrained/'+str(fix_seed)
        random.seed(fix_seed)
        np.random.seed(fix_seed)
        torch.manual_seed(fix_seed)
        torch.cuda.manual_seed(fix_seed)
        torch.cuda.manual_seed_all(fix_seed)
        cudnn.benchmark = False
        cudnn.deterministic = True
        main(args, model_path1, model_path2, model_path3)
