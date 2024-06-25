import os
import numpy as np 
import pandas as pd 

from PIL import Image

import torch
from torchvision import transforms
from torch.utils.data import Dataset


# dataset for OPMD classification with images of two classes stored in csv
class OPMDClassificationDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, img_dir=None):
        self.csv_path = csv_file
        self.csv_file = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.csv_file)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.csv_file.iloc[idx, 0])            

        if 'masked_images' in self.root_dir:
            file_path, file_extension = os.path.splitext(img_name)
            img_name = file_path + "_masked" + file_extension

        image = Image.open(img_name).convert('RGB')
        image = np.asarray(image, dtype=np.float32)
        
        label = self.csv_file.iloc[idx, 1]
        label = np.asarray(label, dtype=np.float32)

        if self.transform is not None:
            image = self.transform(image=image)["image"]

        if self.img_dir is not None:
            fimg_name = os.path.join(self.img_dir, self.csv_file.iloc[idx, 0])      
            fimage = Image.open(fimg_name).convert('RGB')
            fimage = np.asarray(fimage, dtype=np.float32)
            if self.transform is not None:
                fimage = self.transform(image=fimage)["image"]

            ## image = torch.cat((image, fimage), 0)

        return {"image": image, "label": label, "image_name": img_name} 
    

# unit-test the dataset
if __name__ == '__main__':
    root_dir = '../Images'
    csv_file = '../opmd_ind_test.csv'
    
    # test the dataset
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    dataset = OPMDClassificationDataset(csv_file, root_dir, transform=None)
    print(len(dataset))

    print(dataset[0]["image"].shape)
    print(dataset[0]["label"])
