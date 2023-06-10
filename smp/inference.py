import torch 
import os

import albumentations as A
from dataset import XRayInferenceDataset
from torch.utils.data import DataLoader
from torchvision import models
from utils.test import test
import pandas as pd

## 폴더 경로
save_dir = '/opt/ml/input/result/smp/fcn_resnet101_hi'
## pt 파일 이름
model_name = 'fcn_resnet101_hi_epoch1.pt'
model = torch.load(os.path.join(save_dir, model_name))

tf = A.Resize(512, 512)

test_dataset = XRayInferenceDataset(transforms=tf)

test_loader = DataLoader(
    dataset=test_dataset, 
    batch_size=2,
    shuffle=False,
    num_workers=2,
    drop_last=False
)

model = models.segmentation.fcn_resnet101(pretrained=True)

rles, filename_and_class = test(model, test_loader)

classes, filename = zip(*[x.split("_") for x in filename_and_class])

image_name = [os.path.basename(f) for f in filename]

df = pd.DataFrame({
    "image_name": image_name,
    "class": classes,
    "rle": rles,
})

df.to_csv(os.path.join(save_dir, "output.csv"), index=False)