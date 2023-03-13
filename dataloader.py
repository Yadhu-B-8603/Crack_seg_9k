import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import os

class CrackSeg9KDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.img_list = os.listdir(os.path.join(root_dir, 'Images'))
        self.mask_list = os.listdir(os.path.join(root_dir, 'Masks'))
        
    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, 'Images', self.img_list[idx])
        mask_path = os.path.join(self.root_dir, 'Masks', self.mask_list[idx])
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')
        if self.transform is not None:
            image = self.transform(image)
            mask = self.transform(mask)
        return image, mask

# Define the transform you want to apply to the images
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

root_dir = "D:\\iiith intern work\\Dataloader"
img_len = len(os.listdir(os.path.join(root_dir,"Images")))
val_size = int(0.3*img_len)


# Create the dataset and dataloader
crackseg9k_dataset = CrackSeg9KDataset(root_dir, transform=transform)
train,val = torch.utils.data.random_split(crackseg9k_dataset,[img_len - val_size , val_size])
train_loader = DataLoader(train , batch_size = 16 , shuffle = True)
val_loader = DataLoader(val , batch_size = 16 , shuffle = True)
train_features , train_labels = next(iter(train_loader))
val_features , val_labels = next(iter(val_loader))
