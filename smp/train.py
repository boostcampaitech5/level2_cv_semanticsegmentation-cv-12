# python native
import os
# external library
import albumentations as A
# torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, transforms
# visualization
from dataset import XRayDataset
from torch.optim import lr_scheduler
import segmentation_models_pytorch as smp

from utils.base import set_seed
from utils.train import train
from dataset import CLASSES
import wandb

batch_size = 8
lr = 1e-4
random_seed = 21

num_epochs = 200    # CHANGE
val_every = 1 #10

save_dir = "/opt/ml/input/result/smp/"

save_name = 'fcn_resnet101_hi'

if not os.path.isdir(save_dir):                                                           
    os.mkdir(save_dir)

tf = A.Resize(512, 512)


train_dataset = XRayDataset(is_train=True, transforms=tf)
valid_dataset = XRayDataset(is_train=False, transforms=tf)

# image, label = train_dataset[0]

train_loader = DataLoader(
    dataset=train_dataset, 
    batch_size=batch_size,
    shuffle=True,
    num_workers=8,
    drop_last=True,
)

valid_loader = DataLoader(
    dataset=valid_dataset, 
    batch_size=2,
    shuffle=False,
    num_workers=2,
    drop_last=False
)

model = models.segmentation.fcn_resnet101(pretrained=True)

# output class를 data set에 맞도록 수정
model.classifier[4] = nn.Conv2d(512, len(CLASSES), kernel_size=1)

# Loss function 정의
criterion = nn.BCEWithLogitsLoss()

# Optimizer 정의
optimizer = optim.Adam(params=model.parameters(), lr=lr, weight_decay=1e-6)

scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=0.001)

if __name__ == "__main__":
    set_seed(random_seed)

    wandb.login()
    wandb.init(
        project = 'Semantic Segmentation',
        name='smp',
        entity='ganddddi_datacentric',
        # resume= True if args.resume else False
    )

    train(model, 
        train_loader, 
        valid_loader, 
        num_epochs, 
        criterion, 
        optimizer, 
        scheduler, 
        random_seed, 
        val_every, 
        save_dir, 
        save_name)