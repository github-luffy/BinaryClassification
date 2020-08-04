# -*- coding: utf-8 -*-

import os
import cv2
import argparse
from pathlib import Path
import numpy as np

model_path, _ = os.path.split(os.path.realpath(__file__))

def assemble(images_list, train_file, test_file, src_image_path, dst_train_image_path, dst_test_image_path,
             test_nums=1000, label=1, dst_image_name=''):
    idx_keep = np.arange(len(images_list))
    np.random.shuffle(idx_keep)

    with open(test_file, 'a+') as fw:
        for i in range(0, test_nums):
            index = idx_keep[i]
            src_path = os.path.join(src_image_path, images_list[index])
            dst_path = os.path.join(dst_test_image_path, dst_image_name % i)
            cv2.imwrite(dst_path, cv2.imread(src_path))
            fw.write(dst_path + ' ' + str(label) + '\n')

    with open(train_file, 'a+') as fw:
        for i in range(test_nums, len(idx_keep)):
            index = idx_keep[i]
            src_path = os.path.join(src_image_path, images_list[index])
            dst_path = os.path.join(dst_train_image_path, dst_image_name % i)
            cv2.imwrite(dst_path, cv2.imread(src_path))
            fw.write(dst_path + ' ' + str(label) + '\n')


def main(args):
    print(model_path)
    data = {'negative': args.nums_test_neg, 'positive':args.nums_test_pos}
    src_open_image_path = os.path.join(model_path, 'positive')
    src_close_image_path = os.path.join(model_path, 'negative')
    open_images = os.listdir(src_open_image_path)
    close_images = os.listdir(src_close_image_path)

    assert len(open_images) > args.nums_test_pos
    assert len(close_images) > args.nums_test_neg
    assert len(open_images) != 0 or len(close_images) != 0

    dst_train_open_image_path = os.path.join(model_path, 'train_data/positive')
    dst_train_close_image_path = os.path.join(model_path, 'train_data/negative')

    dst_test_open_image_path = os.path.join(model_path, 'test_data/positive')
    dst_test_close_image_path = os.path.join(model_path, 'test_data/negative')
    
    cp1 = Path(dst_train_open_image_path)
    cp1.mkdir(exist_ok=True, parents=True)
    
    cp2 = Path(dst_train_close_image_path)
    cp2.mkdir(exist_ok=True, parents=True)
    
    cp3 = Path(dst_test_open_image_path)
    cp3.mkdir(exist_ok=True, parents=True)
    
    cp4 = Path(dst_test_close_image_path)
    cp4.mkdir(exist_ok=True, parents=True)

    train_file = os.path.join(model_path, 'train_data/list.txt')
    test_file = os.path.join(model_path, 'test_data/list.txt')

    assemble(open_images, train_file, test_file, src_open_image_path, dst_train_open_image_path,
             dst_test_open_image_path, test_nums=data['positive'], label=1, dst_image_name='positive_%08d.jpg')

    assemble(close_images, train_file, test_file, src_close_image_path, dst_train_close_image_path,
             dst_test_close_image_path, test_nums=data['negative'], label=0, dst_image_name='negative_%08d.jpg')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--nums_test_pos', type=int, default=1000)
    parser.add_argument('--nums_test_neg', type=int, default=1000)

    args = parser.parse_args()
    main(args)

