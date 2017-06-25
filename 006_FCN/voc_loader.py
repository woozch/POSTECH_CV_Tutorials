# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Routine for decoding the CIFAR-10 binary file format."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle
import numpy as np
import PIL
from PIL import Image
from scipy.misc import imresize

CUR_PATH = os.getcwd()
VGG_MEAN = np.array([103.939, 116.779, 123.68])
# Global constants describing the CIFAR-10 data set.
CLASS_NAMES = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 
    'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 
    'person', 'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor'
    ]

class VOC_loader():
    def __init__(self, params):
        """ Initialize CIFAR-10 loader """

        self.num_classes = params['num_classes']
        self.image_size = params['image_size']
        self.split_root = params['split_root']
        self.image_root = params['image_root'] #'VOCdevkit/VOC2012/JPEGImages'
        self.segmap_root = params['segmap_root'] #'VOCdevkit/VOC2012/SegmentationClass'
        
        # load training data
        train_list = os.path.join(self.split_root, 'train.txt')
        with open(train_list, 'r') as file:
           self.train_files = file.readlines()
        self.train_files = [l.strip() for l in self.train_files]

        # load testing data
        test_list = os.path.join(self.split_root, 'val.txt')
        with open(train_list, 'r') as file:
           self.test_files = file.readlines()
        self.test_files = [l.strip() for l in self.test_files]

        # maintain statistics
        self.num_data = {
                'train': len(self.train_files),
                'test': len(self.test_files),
            }

        # setting iterator
        self.iterator = {
                'train':0,
                'test':0,
            }
        self.test_wrapped = False

        # pallete
        self.palette = None
        
    def _increment_iterator(self, batch_size, split):
        self.iterator[split] += 1

        if self.iterator[split] == self.num_data[split]:
            self.iterator[split] = 0
            if split == 'test':
                self.test_wrapped = True
    
    def _prepro(self, image, seg_label, random_crop=False, flip=False):
        
        img_size = self.image_size
        if random_crop:
            # Resize the image or segmentation to bigger size than image size
            img_size = 256 if img_size == 224 else 480
            image = image.resize((img_size,img_size))
            seg_label = seg_label.resize((img_size,img_size), resample=PIL.Image.NEAREST)

            # TODO: crop the region

        else:
            image = image.resize((img_size,img_size))
            seg_label = seg_label.resize((img_size,img_size), resample=PIL.Image.NEAREST)

        if flip:
            # TODO
            TODO = True

        # return image, seg_map, seg_label
        return np.asarray(image, dtype=np.float), \
            np.asarray(seg_label.convert('RGB'), dtype=np.uint8), \
            np.asarray(seg_label, dtype=np.uint8)
    
    def get_num_train_examples(self):
        return self.num_data['train']
    
    def get_num_test_examples(self):
        return self.num_data['test']
    
    def get_image_size(self):
        return self.image_size
    
    def get_num_classes(self):
        return self.num_classes
    
    def get_class_names(self):
        return CLASS_NAMES

    def get_palette(self):
        return self.palette

    def reset(self):
        self.iterator['train'] = 0
        self.iterator['test'] = 0
        self.test_wrapped = False
        
    def get_batch(self, batch_size, split='train', random_crop=False, flip=False):
        """ Get batch data """
        batch = {}
        
        if split == 'test' and self.test_wrapped:    
            batch['wrapped'] = self.test_wrapped
            return batch

        # extract images and labels of batch size
        batch_images = np.zeros((batch_size, self.image_size, self.image_size, 3), dtype=np.float)
        batch_seg_maps = []
        batch_seg_labels = np.zeros((batch_size, self.image_size, self.image_size), 
                dtype=np.uint8)

        it = self.iterator[split]
        for bi in range(batch_size):
            # load image and segmentation map
            img = Image.open(os.path.join(self.image_root, self.train_files[it] + '.jpg'))
            seg_label = Image.open(os.path.join(self.segmap_root, self.train_files[it] + '.png'))

            if self.palette is None:
                self.palette = seg_label.getpalette()
            
            # preprocessing them
            batch_images[bi], seg_map, batch_seg_labels[bi] = \
                self._prepro(img, seg_label, random_crop, flip)
            batch_seg_maps.append(seg_map)

            # increment the iterator
            self._increment_iterator(batch_size, split)

        # change index of the boundary label from 255 to 21
        np.place(batch_seg_labels, batch_seg_labels==255, 21)
        batch['images'] = batch_images
        batch['seg_maps'] = batch_seg_maps
        batch['seg_labels'] = batch_seg_labels
        if split == 'test': batch['wrapped'] = self.test_wrapped
        
        return batch
