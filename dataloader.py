#!/usr/bin/env python
# coding: utf-8

# In[23]:


import torch 
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
#import cv2
from skimage import io
import matplotlib.pyplot as plt
from PIL import Image

img_dir = "D:\\iiith intern work\\Dataloader\\Images"


class cracksegDataset(torch.utils.data.Dataset):
    def __init__(self , img_dir , transform , target_transform):
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
    
    def __len__(self):
        count = 0
        for path in os.listdir(img_dir):
            if os.path.isfile(os.path.join(img_dir , path)):
                count += 1
        return count
    
    def __getitem__(self , idx):
        img_name_list = os.listdir(self.img_dir)
        img_label = img_name_list[idx]
        img_path = os.path.join(self.img_dir , img_label)
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)
            
        return image , img_label

class LabelTransform:
    def __init__(self):
        pass
    
    def __call__(self , label):
        return label

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p = 0.25),
    transforms.ToTensor(),
])

target_transforms = LabelTransform()

dataset = cracksegDataset(img_dir = "D:\iiith intern work\Dataloader\Images",
                          transform = transform,
                          target_transform = target_transforms)
                          
train , val = torch.utils.data.random_split(dataset , [2850 , 6645] )
train_loader= DataLoader(train, batch_size = 16 , shuffle = True )
val_loader = DataLoader(val , batch_size = 16 , shuffle = True)
train_features , train_labels = next(iter(train_loader))
val_features , val_labels = next(iter(val_loader))


# In[ ]:




