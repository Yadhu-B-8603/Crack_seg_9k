import torch 
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
from skimage import io

# This is the original image directory
img_dir = "D:\iiith intern work\Images"

#custom dataset
class cracksegDataset(torch.utils.data.Dataset):
    def __init__(self , csv_file , img_dir , transform = None , target_transform = None):
        self.csv_file = csv_file
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
    
    def __len__(self):
        return len(self.csv_file)
    
    def __getitem__(self , idx):
        img_path = os.path.join(self.img_dir , self.csv_file.iloc[idx , 0])
        image = io.imread(img_path)
        y_label = torch.tensor(self.csv_file.iloc[idx , 1])
        
        if self.transform:
            image = self.transform(image)
            

        return [image , y_label]    

# dataloader
dataset = cracksegDataset(csv_file = "D:\iiith intern work\meta_data.csv", 
                          img_dir = "D:\iiith intern work\Images",
                          transform = transforms.ToTensor())
train_loader = DataLoader(dataset = img_dir , batch_size = 16 , shuffle = True)
