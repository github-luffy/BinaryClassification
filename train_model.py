# -*- coding: UTF-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
print('pid: {}     GPU: {}'.format(os.getpid(), os.environ['CUDA_VISIBLE_DEVICES']))

import torch
import torchvision as tv
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import numpy as np
import cv2
import argparse
import sys
import time
from dataset import DataSet
from model import MobileNetV2, ONet, PNet, BlazeLandMark


def main(args):
    print(args)
    np.random.seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_data_transforms = tv.transforms.Compose([
        tv.transforms.Resize((args.image_size, args.image_size)),
        tv.transforms.RandomCrop(args.image_size, 4),
        tv.transforms.RandomHorizontalFlip(),
        tv.transforms.RandomRotation((-45, 45)),
        tv.transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    test_data_transforms = tv.transforms.Compose([
        tv.transforms.Resize((args.image_size, args.image_size)),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    train_dataset = DataSet(args.file_list, args.image_channels, args.image_size, transforms=train_data_transforms,
                            is_train=True)
    test_dataset = DataSet(args.test_list, args.image_channels, args.image_size, transforms=test_data_transforms,
                           is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=12)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=12)

    model_dir = args.model_dir
    print(model_dir)
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    print('Total number of examples: {}'.format(len(train_dataset)))
    print('Test number of examples: {}'.format(len(test_dataset)))
    print('Model dir: {}'.format(model_dir))

    # model = ONet(nums_class=args.nums_class)
    # model = PNet(nums_class=args.nums_class)
    model = MobileNetV2(input_channels=3, nums_class=args.nums_class)
    # model = BlazeLandMark(nums_class=args.nums_class)
    
    if args.pretrained_model:
        pretrained_model = args.pretrained_model
        if args.all_model:
            print('load all model, model graph and weight included!')
            if not os.path.isdir(pretrained_model):
                print('Restoring pretrained model: {}'.format(pretrained_model))
                model = torch.load(pretrained_model)
            else:
                print('Model directory: {}'.format(pretrained_model))
                files = os.listdir(pretrained_model)
                assert len(files) == 1 and files[0].split('.')[-1] in ['pt', 'pth']
                model_path = os.path.join(pretrained_model, files[0])
                print('Model name:{}'.format(files[0]))
                model = torch.load(model_path)
        else:
            if not os.path.isdir(pretrained_model):
                print('Restoring pretrained model: {}'.format(pretrained_model))
                model.load_state_dict(torch.load(pretrained_model))
            else:
                print('Model directory: {}'.format(pretrained_model))
                files = os.listdir(pretrained_model)
                assert len(files) == 1 and files[0].split('.')[-1] in ['pt', 'pth']
                model_path = os.path.join(pretrained_model, files[0])
                print('Model name:{}'.format(files[0]))
                model.load_state_dict(torch.load(model_path))
        test(test_loader, model, args, device)

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)  # optimizer
    lr_epoch = args.lr_epoch.strip().split(',')
    lr_epoch = list(map(int, lr_epoch))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_epoch, gamma=0.1)
    criterion = nn.CrossEntropyLoss()

    print('Running train.')
    start_time = time.time()
    for epoch in range(args.max_epoch):
        model.train()
        scheduler.step(epoch)

        correct = 0.
        total = 0
        for i_batch, (images_batch, labels_batch) in enumerate(train_loader):
            images_batch = Variable(images_batch.to(device))
            labels_batch = Variable(labels_batch.to(device))

            # labels_batch = torch.zeros(args.batch_size, args.nums_class).scatter_(1, labels_batch, 1)
            # labels_batch = labels_batch.to(device, dtype=torch.int64)

            pre_labels = model(images_batch)
            #print('...............',pre_labels)
            print(pre_labels.size(), labels_batch.size(), labels_batch.squeeze())
            loss = criterion(pre_labels, labels_batch.squeeze(axis=1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if ((i_batch + 1) % 100) == 0 or (i_batch + 1) == len(train_loader):
                Epoch = 'Epoch:[{:<4}][{:<4}/{:<4}]'.format(epoch, i_batch + 1, len(train_loader))
                Loss = 'Loss: {:2.3f}'.format(loss.item())
                trained_sum_iters = len(train_loader) * epoch + i_batch + 1
                average_time = (time.time() - start_time) / trained_sum_iters
                remain_time = average_time * (len(train_loader) * args.max_epoch - trained_sum_iters) / 3600
                print('{}\t{}\t lr {:2.3}\t average_time:{:.3f}s\t remain_time:{:.3f}h'.format(Epoch, Loss,
                                                                                               optimizer.param_groups[0]['lr'],
                                                                                               average_time,
                                                                                               remain_time))
            _, predicted = torch.max(pre_labels.data, 1)

            total += labels_batch.size(0)

            correct += (predicted == labels_batch[:, 0]).sum()

        # save model
        checkpoint_path = os.path.join(model_dir, 'model_'+str(epoch)+'.pth')
        if args.all_model:
            torch.save(model, checkpoint_path)
        else:
            torch.save(model.state_dict(), checkpoint_path)
        print('Train Images: {}, right nums: {}, right rate:{:.3f}%'.format(total, correct, correct.to(torch.float32) * 100 / total))
        test(test_loader, model, device)


def test(test_loader, model, device):

    model.eval()

    correct = 0.
    total = 0

    for i_batch, (images_batch, labels_batch) in enumerate(test_loader):
        images_batch = images_batch.to(device)
        labels_batch = labels_batch.to(device)
        pre_labels = model(images_batch)
        _, predicted = torch.max(pre_labels.data, 1)

        total += labels_batch.size(0)

        correct += (predicted == labels_batch[:, 0]).sum()
    print('Test Images: {}, right nums: {}, right rate:{:.3f}%'.format(total, correct, correct.to(torch.float32) * 100 / total))



def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--file_list', type=str, default='./data/train_data/list.txt')
    parser.add_argument('--test_list', type=str, default='./data/test_data/list.txt') #公司采集
    parser.add_argument('--loss_log_dir', type=str, default='./train_loss_log/')
    parser.add_argument('--seed', type=int, default=666)
    parser.add_argument('--nums_class', type=int, default=2)
    parser.add_argument('--max_epoch', type=int, default=500)
    parser.add_argument('--image_size', type=int, default=32)
    parser.add_argument('--image_channels', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--pretrained_model', type=str, default='')
    parser.add_argument('--model_dir', type=str, default='./pretrained_model')#公司采集
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--lr_epoch', type=str, default='10,20,50,100,200,500')
    parser.add_argument('--weight_decay', type=float, default=5e-5)
    parser.add_argument('--level', type=str, default='L5')
    # parser.add_argument('--save_image_example', action='store_false', default=True)
    parser.add_argument('--all_model', action='store_true', default=True)

    return parser.parse_args(argv)


if __name__ == '__main__':
    print(sys.argv)
    main(parse_arguments(sys.argv[1:]))
