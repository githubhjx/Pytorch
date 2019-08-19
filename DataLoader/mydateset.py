#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019-08-08 16:43
# @Author  : Harry


import os
import torch
import cv2
import glob
from torch.utils import data
from torchvision.datasets import ImageFolder


path = '/Users/it00002807/PycharmProjects/Lab/train_data/'
nir_path = '/Users/it00002807/PycharmProjects/Lab/train_data/NIR/'
vis_path = '/Users/it00002807/PycharmProjects/Lab/train_data/VIS/'
gallery_path = '/Users/it00002807/PycharmProjects/Lab/train_data/VIS/'
probe_path = '/Users/it00002807/PycharmProjects/Lab/train_data/NIR/'


class MyDatasets(data.Dataset):
    """Definition My NIR-VIS Dataset"""

    def __init__(self, nir_path, vis_path, gallery_path, probe_path, mode):
        """Initialize Parameters"""
        self.mode = mode
        self.nir_path = nir_path
        self.vis_path = vis_path
        self.gallery_path = gallery_path
        self.probe_path = probe_path
        self.train_dataset = []
        self.test_dataset = {'Gallery':[], 'Probe':[]}
        self.read_img()

        if mode == 'train':
            self.num_images = len(self.train_dataset)
        else:
            self.num_images = len(self.test_dataset)

    def read_img(self):
        """Read Image and Simple Pretreatment"""
        cate = [self.nir_path + x for x in os.listdir(self.nir_path) if os.path.isdir(self.nir_path + x)]
        cate.sort()

        cate1 = [self.vis_path + x for x in os.listdir(self.vis_path) if os.path.isdir(self.vis_path + x)]
        cate1.sort()

        cate2 = [self.gallery_path + x for x in os.listdir(self.gallery_path) if os.path.isdir(self.gallery_path + x)]
        cate2.sort()

        cate3 = [self.probe_path + x for x in os.listdir(self.probe_path) if os.path.isdir(self.probe_path + x)]
        cate3.sort()

        nir_imgs = []
        nir_labels = []
        vis_imgs = []
        vis_labels = []

        gallery_imgs = []
        gallery_labels = []
        probe_imgs = []
        probe_labels = []

        for idx, folder in enumerate(cate):
            temp = glob.glob(folder + '/*.bmp')
            nir_imgs.extend(temp)
            nir_labels.extend(len(temp)*[idx])

        for idx, folder in enumerate(cate1):
            temp = glob.glob(folder + '/*.jpg')
            vis_imgs.extend(temp)
            vis_labels.extend(len(temp)*[idx])

        # load image and label dataset
        for i in range(len(nir_imgs)):
            self.train_dataset.extend([nir_imgs[i], nir_labels[i], vis_imgs[i], vis_labels[i]])

        for idx, folder in enumerate(cate2):
            temp = glob.glob(folder + '/*.jpg')
            gallery_imgs.extend(temp)
            gallery_labels.extend(len(temp)*[idx])

        for idx, folder in enumerate(cate1):
            temp = glob.glob(folder + '/*.bmp')
            probe_imgs.extend(temp)
            probe_labels.extend(len(temp)*[idx])

        # load image and label dataset
        for i in range(len(gallery_imgs)):
            self.test_dataset['Gallery'].extend([gallery_imgs[i], gallery_labels[i]])

        for i in range(len(probe_imgs)):
            self.test_dataset['Probe'].extend([probe_imgs[i], probe_labels[i]])

        print('The Data has be read success!')

    def __getitem__(self, index):
        """Return one image and its corresponding attribute label."""
        if self.mode == 'train':
            dataset = self.train_dataset

            nir_file, nir_label, vis_file, vis_label = dataset[index]

            return self.transform(nir_file), torch.FloatTensor(nir_label), self.transform(vis_file), torch.FloatTensor(vis_label)

    def __len__(self):
        """Return the number of images."""
        return self.num_images


dataset = MyDatasets(nir_path, vis_path, 'train')
dataset.read_img()
