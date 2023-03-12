import os
import sys
import time
import glob
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import genotypes
import torch.utils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
import torch.utils.data as dataloader

from torch.autograd import Variable
from model import NetworkCIFAR as Network
# from Dataset import MyData
# from thop import profile
import matplotlib.pyplot as plt
# import random
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split

df = pd.DataFrame(columns=['train Loss', 'training accuracy', ' valid Loss', 'validing accuracy'])
df.to_csv("evaluation/result_lora_4cell.csv", index=False)
parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=100, help='report frequency')
parser.add_argument('--gpu', type=int, default=2, help='gpu device id')
parser.add_argument('--epochs', type=int, default=100, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=36, help='num of init channels')
parser.add_argument('--layers', type=int, default=4, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--arch', type=str, default='DARTS', help='which architecture to use')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.7, help='portion of training data')
args = parser.parse_args()

# args.save = 'eval-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
# utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler('./evaluation/log.txt')
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)
CIFAR_CLASSES = 30


def main():
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    np.random.seed(args.seed)
    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    logging.info('gpu device = %d' % args.gpu)
    logging.info("args = %s", args)

    genotype = eval("genotypes.%s" % args.arch)
    model = Network(args.init_channels, CIFAR_CLASSES, args.layers, args.auxiliary, genotype)
    model = model.cuda()

    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))
    # input = torch.cuda.FloatTensor(1, 2, 100, 60)
    # flops, params = profile(model, inputs=(input,))
    # print(flops)
    # cross entropy loss
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    optimizer = torch.optim.SGD(
        model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )


    data = np.load('LoRa_RFF/dataset/Train/X_train_30class.npy')
    label = np.load('LoRa_RFF/dataset/Train/Y_train_30class.npy')
    # data = data.reshape(-1, 1, 102, 62)

    # Split the dasetset into validation and training sets.
    data_train, data_valid, label_train, label_valid = train_test_split(data, label, test_size=0.3, shuffle=True)

    del data, label
    data_train = torch.from_numpy(data_train)
    data_train = data_train.type(torch.FloatTensor)
    label_train = torch.from_numpy(label_train)
    label_train = label_train.type(torch.LongTensor)
    label_train = label_train.squeeze()

    data_valid = torch.from_numpy(data_valid)
    data_valid = data_valid.type(torch.FloatTensor)
    label_valid = torch.from_numpy(label_valid)
    label_valid = label_valid.type(torch.LongTensor)
    label_valid = label_valid.squeeze()

    train_data = torch.utils.data.TensorDataset(data_train, label_train)
    valid_data = torch.utils.data.TensorDataset(data_valid, label_valid)

    train_queue = dataloader.DataLoader(
        dataset=train_data,
        batch_size=args.batch_size,
        shuffle=True,
    )
    valid_queue = dataloader.DataLoader(
        dataset=valid_data,
        batch_size=args.batch_size,
        shuffle=False,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs))

    tra_acc = []
    tra_loss = []
    val_acc = []
    val_loss = []
    for epoch in range(args.epochs):
        scheduler.step()
        logging.info('epoch %d lr %e', epoch, scheduler.get_lr()[0])
        model.drop_path_prob = args.drop_path_prob * epoch / args.epochs

        train_acc, train_obj = train(train_queue, model, criterion, optimizer)
        logging.info('train_acc %f', train_acc)
        tra_acc.append(train_acc)
        tra_loss.append(train_obj)

        valid_acc, valid_obj = infer(valid_queue, model, criterion)
        logging.info('valid_acc %f', valid_acc)
        val_acc.append(valid_acc)
        val_loss.append(valid_obj)
        list_acc = [train_obj, train_acc, valid_obj, valid_acc]
        data = pd.DataFrame([list_acc])
        data.to_csv('evaluation/result_lora_4cell.csv', mode='a', header=False, index=False)

        # utils.save(model, os.path.join(args.save, 'weights.pt'))
        # utils.save(model, os.path.join('F:/ADS-B/evaluation', 'weights.pt'))
        utils.save(model, 'evaluation/weight_lora_4cell.pt')
    #
    # plt.plot(np.arange(len(tra_loss)), tra_loss, label="train loss")
    #
    # plt.plot(np.arange(len(tra_acc)), tra_acc, label="train acc")
    #
    # plt.plot(np.arange(len(val_loss)), val_loss, label="valid loss")
    #
    # plt.plot(np.arange(len(val_acc)), val_acc, label="valid acc")
    # plt.legend()
    # plt.xlabel('Epoch')
    # plt.ylabel("Accuracy and loss")
    # plt.title('Model accuracy&loss')
    plt.show()


def train(train_queue, model, criterion, optimizer):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.train()

    for step, (input, target) in enumerate(train_queue):
        input = Variable(input).cuda()
        target = Variable(target).cuda()

        optimizer.zero_grad()
        logits, logits_aux = model(input)
        loss = criterion(logits, target)
        if args.auxiliary:
            loss_aux = criterion(logits_aux, target)
            loss += args.auxiliary_weight * loss_aux
        loss.backward()
        nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)
        optimizer.step()

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 2))
        n = input.size(0)
        # objs.update(loss.data[0], n)
        # top1.update(prec1.data[0], n)
        # top5.update(prec5.data[0], n)
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

        if step % args.report_freq == 0:
            logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg


def infer(valid_queue, model, criterion):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.eval()

    for step, (input, target) in enumerate(valid_queue):
        input = Variable(input, volatile=True).cuda()
        target = Variable(target, volatile=True).cuda()

        logits, _ = model(input)
        loss = criterion(logits, target)

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 2))
        n = input.size(0)
        # objs.update(loss.data[0], n)
        # top1.update(prec1.data[0], n)
        # top5.update(prec5.data[0], n)
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

        if step % args.report_freq == 0:
            logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg


if __name__ == '__main__':
    main()
