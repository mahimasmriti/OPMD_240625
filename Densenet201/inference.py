import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import os
import numpy as np

import albumentations as A
from albumentations.pytorch import ToTensorV2

from monai.transforms import Activations, AsDiscrete
from sklearn.metrics import classification_report, confusion_matrix

from tqdm import tqdm

from models.densenet import DenseNet201
from dataset2 import OPMDClassificationDataset
from utils import get_config, set_seed, calculate_metrics

from torchsummary import summary


config = get_config('config.yaml')


set_seed(config['seed'])

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'

test_transforms = A.Compose([
            A.Resize(config['resize'], config['resize']),
            A.CenterCrop(config['crop'], config['crop']),
            ToTensorV2(),
        ])

def get_model_size(model):
    num_params = sum(p.numel() for p in model.parameters())
    num_buffers = sum(b.numel() for b in model.buffers())
    total_size = num_params * 4 + num_buffers * 4  # Assuming 32-bit floating point
    return total_size

test_data_csv = '../opmd_ind_test.csv'
output_csv = './out_opmd_ind_test.csv'

imgs_dir = '../Images'

test_dataset = OPMDClassificationDataset(test_data_csv, imgs_dir, transform=test_transforms)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

model = DenseNet201(num_classes=config['num_classes'], quantize=config['quantize'])
# print (model)
total_params = sum(p.numel() for p in model.parameters())
print('Total number of prameters',total_params)
print('Model size', get_model_size(model))

model = DenseNet201(num_classes=config['num_classes'], quantize=config['quantize']).to(device)

model = nn.DataParallel(model)

model.load_state_dict(torch.load('./densenet201_nchannel_3_lr_0.0001_bs_32_epochs_100/model_52.pth')['model_state_dict'])

model = model.module.to(device)

model.eval()

img_name_list = []
labels_list = []
outputs_list = []
outputs_probs_list = []  

# test the model

with torch.no_grad():
    for batch in tqdm(test_dataloader, total=len(test_dataloader)):
        images = batch['image'].to(device)
        labels = batch['label'].to(device)

        img_name = batch['image_name']
        img_name_list.append(img_name)

        outputs = model(images)
        outputs = outputs.squeeze()

        outputs_probs = Activations(sigmoid=True)(outputs)
        outputs = AsDiscrete(threshold=0.5)(outputs_probs)
        outputs = outputs.squeeze()

        labels_list.append(labels.tolist())
        outputs_list.append(outputs.tolist())
        outputs_probs_list.append(outputs_probs.tolist())
        
    print('Classification report:')
    print(classification_report(np.asarray(labels_list), np.asarray(outputs_list)))

    print('Confusion matrix:')
    print(confusion_matrix(np.asarray(labels_list), np.asarray(outputs_list)))

    precision, recall, sensitivity, specificity, f1, accuracy = calculate_metrics(labels_list, outputs_list)

    print('Test metrics:')
    print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, Sensitivity: {sensitivity:.4f}, Specificity: {specificity:.4f}, F1: {f1:.4f}, Accuracy: {accuracy:.4f}\n')

    torch.save(model.state_dict(), "temp.p")
    print('Size of the model(MB):', os.path.getsize("temp.p")/1e6)
    os.remove('temp.p')

# store each image's name, label and output
with open(output_csv, 'w') as f:
    f.write('image_name,label,output,output_probs\n')
    for i in range(len(img_name_list)):
        f.write(f'{img_name_list[i]},{labels_list[i]},{outputs_list[i]},{outputs_probs_list[i]}\n')
