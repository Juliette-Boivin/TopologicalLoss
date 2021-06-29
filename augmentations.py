#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import albumentations as A
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import random

transform = A.Compose([
    A.RandomCrop(width=256, height=256, p=0.1),
    A.HorizontalFlip(p=0.2),
    A.geometric.rotate.Rotate (limit=random.randint(0, 90), p=0.5),
    A.geometric.transforms.ElasticTransform(p=0.7),
    A.geometric.transforms.ShiftScaleRotate(p=0.5),
    A.transforms.OpticalDistortion(p=0.7),
    A.transforms.RandomBrightnessContrast(p=0.3),
    
])


path_ = '/home/jboivin/Documents/CHASEDB1'
elem_list = os.listdir(path_ + '/Images')
path = '/home/jboivin/Documents/CHASEDB1_augmented'

z = 982
for i in elem_list:
    image = cv2.imread(path + '/Images/' + i)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    mask_1 = cv2.imread(path + '/1stHO/' + i.replace('jpg', 'png'))
    mask_2 = cv2.imread(path + '/2ndHO/' + i.replace('jpg', 'png'))
    masks = [mask_1, mask_2]

    transformed = transform(image=image, masks=masks)
    transformed_image = transformed['image']
    transformed_mask = transformed['masks']
    
    im = Image.fromarray(transformed_image)
    im.save(path + '/Images/' + str(z) + '.jpg')
    
    msk1 = np.array(transformed_mask)[0, :, :, 0]
    msk_1 = Image.fromarray(msk1)
    msk_1.save(path + '/1stHO/' + str(z) + '.png')
    
    msk2 = np.array(transformed_mask)[1, :, :, 0]
    msk_2 = Image.fromarray(msk2)
    msk_2.save(path + '/2ndHO/' + str(z) + '.png')
    
    z += 1
    
    
