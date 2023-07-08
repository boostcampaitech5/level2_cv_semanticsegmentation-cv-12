import os

import albumentations as A
import hydra
import lightning as L
import torch.nn as nn
from hydra.utils import instantiate
from models.base_module import Module
from omegaconf import DictConfig
from torch.utils.data import Dataset, DataLoader
import numpy as np

import torch
import tqdm
import torch
import random
import time

import cv2
import pandas as pd
import torch.nn.functional as F
from data.data_module import CLASSES, CLASS2IND, IND2CLASS

import matplotlib.pyplot as plt
from models.base_module import label2rgb
# hyperparamters
IMAGE_ROOT = "/opt/ml/input/data/test/DCM"
# PATH = '/opt/ml/input/code/smp/checkpoints/hm_unet++-efficientB7-adam-cosineLR-1024/best.ckpt'
# PATH = '/opt/ml/input/code/smp/checkpoints/ar_Unet++-maxvit-adamw-focal_dice_aug1-randomcrop-1024-edit_train_valid/best.ckpt'
PATH = '/opt/ml/input/code/smp/checkpoints/hm_unet++-efficientB7-adam-cosinelr-foldN/fold4_best.ckpt'
PATH_SUBCLASS = '/opt/ml/input/code/smp/checkpoints/sw_unet++-adam-cosinelr-efficient_b7_class8-512-all_fold/fold4_best.ckpt'
RESULT = 'output_softensemble.csv'
resolution = 1024

RANDOM_SEED = 21

CLASSES = [
    'finger-1', 'finger-2', 'finger-3', 'finger-4', 'finger-5',
    'finger-6', 'finger-7', 'finger-8', 'finger-9', 'finger-10',
    'finger-11', 'finger-12', 'finger-13', 'finger-14', 'finger-15',
    'finger-16', 'finger-17', 'finger-18', 'finger-19', 'Trapezium',
    'Trapezoid', 'Capitate', 'Hamate', 'Scaphoid', 'Lunate',
    'Triquetrum', 'Pisiform', 'Radius', 'Ulna',
]

CLASS2IND = {v: i for i, v in enumerate(CLASSES)}
IND2CLASS = {v: k for k, v in CLASS2IND.items()}

pngs = {
    os.path.relpath(os.path.join(root, fname), start=IMAGE_ROOT)
    for root, _dirs, files in os.walk(IMAGE_ROOT)
    for fname in files
    if os.path.splitext(fname)[1].lower() == ".png"
}

def set_seed():
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

# mask map으로 나오는 인퍼런스 결과를 RLE로 인코딩 합니다.
def encode_mask_to_rle(mask):
    '''
    mask: numpy array binary mask 
    1 - mask 
    0 - background
    Returns encoded run length 
    '''
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

# RLE로 인코딩된 결과를 mask map으로 복원합니다.
def decode_rle_to_mask(rle, height, width):
    s = rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(height * width, dtype=np.uint8)
    
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    
    return img.reshape(height, width)

class XRayInferenceDataset(Dataset):
    def __init__(self, transforms=None):
        _filenames = pngs
        _filenames = np.array(sorted(_filenames))
        
        self.filenames = _filenames
        self.transforms = transforms
    
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, item):
        image_name = self.filenames[item]
        image_path = os.path.join(IMAGE_ROOT, image_name)
        
        image = cv2.imread(image_path)
        image = image / 255.
        
        if self.transforms is not None:
            inputs = {"image": image}
            result = self.transforms(**inputs)
            image = result["image"]

        # to tenser will be done later
        image = image.transpose(2, 0, 1)    # make channel first
        
        image = torch.from_numpy(image).float()
            
        return image, image_name

def ensemble_with_lessclassmodel(model, model2, data_loader, thr=0.5):
    set_seed()
    model = model.cuda()
    model2 = model2.cuda()
    model.eval()
    model2.eval()

    rles = []
    filename_and_class = []
    with torch.no_grad():
        n_class = len(CLASSES)

        # for step, (images, image_names) in tqdm(enumerate(data_loader), total=len(data_loader)):
        for step, (images, image_names) in enumerate(data_loader):
            images = images.cuda()    
            outputs = model(images)
            outputs2 = model2(images)
            # restore original size
            outputs = F.interpolate(outputs, size=(2048, 2048), mode="bilinear")
            outputs = torch.sigmoid(outputs)
            outputs2 = F.interpolate(outputs2, size=(2048, 2048), mode="bilinear")
            outputs2 = torch.sigmoid(outputs2)
            x = 829
            y = 1487
            # print('outputs before', outputs[0, 20, y:y+10, x:x+10])
            # print('outputs2 before', outputs2[0, 1, y:y+10, x:x+10])
            # outputs[:, 19:, :, :] = (outputs[:, 19:, :, :]*0.3 + outputs2*0.7) ## 10 class
            outputs[:, 19:27, :, :] = (outputs[:, 19:27, :, :]*0.5 + outputs2*0.5) ## 8 class
            # print('outputs after', outputs[0, 20, y:y+10, x:x+10])

            outputs = (outputs > thr).detach().cpu().numpy()
            # exit()
            for output, image_name in zip(outputs, image_names):
                for c, segm in enumerate(output):
                    rle = encode_mask_to_rle(segm)
                    rles.append(rle)
                    filename_and_class.append(f"{IND2CLASS[c]}_{image_name}")
            
    return rles, filename_and_class

def ensemble_with_lessclassmodel_crop512(model, model2, data_loader, data_loader_crop512, thr=0.5):
    set_seed()
    model = model.cuda()
    model2 = model2.cuda()
    model.eval()
    model2.eval()

    rles = []
    filename_and_class = []
    with torch.no_grad():
        n_class = len(CLASSES)

        # for step, (images, image_names) in tqdm(enumerate(data_loader), total=len(data_loader)):
        for step, ((images, image_names), (images_crop512, image_names2)) in enumerate(zip(data_loader, data_loader_crop512)):
            images = images.cuda()
            images_crop512 = images_crop512.cuda()

            outputs = model(images)
            outputs2 = model2(images_crop512)
            outputs = F.interpolate(outputs, size=(2048, 2048), mode="bilinear")
            outputs = torch.sigmoid(outputs)
            outputs2 = torch.sigmoid(outputs2)
            x = 829
            y = 1487

            x_min = 724
            y_min = 1200
            x_max = 1236
            y_max = 1712
            # print('outputs before', outputs[0, 20, y:y+10, x:x+10])
            # print('outputs2 before', outputs2[0, 1, y-y_min:y+10-y_min, x-x_min:x+10-x_min])
            outputs[:, 19:27, y_min:y_max, x_min:x_max] = (outputs[:, 19:27, y_min:y_max, x_min:x_max]*0.5 + outputs2*0.5) ## 8 class

            # print('outputs after', outputs[0, 20, y:y+10, x:x+10])
            # exit()
            outputs = (outputs > thr).detach().cpu().numpy()
            for output, image_name in zip(outputs, image_names):
                for c, segm in enumerate(output):
                    rle = encode_mask_to_rle(segm)
                    rles.append(rle)
                    filename_and_class.append(f"{IND2CLASS[c]}_{image_name}")
            
    return rles, filename_and_class

def ensemble_with_patch_inOneImage(model, data_loader, thr=0.5):
    set_seed()
    model = model.cuda()
    model.eval()

    rles = []
    filename_and_class = []
    with torch.no_grad():
        n_class = len(CLASSES)

        # for step, (images, image_names) in tqdm(enumerate(data_loader), total=len(data_loader)):
        for step, (images, image_names) in enumerate(data_loader):
            images = images.cuda()
            n_stride = 3
            
            outputs = torch.zeros((images.shape[0], 29, 2048, 2048)).cuda()
            sum_num = torch.zeros((2048, 2048)).cuda()
            for i in range(n_stride):
                for j in range(n_stride):
                    ii = i*2048//(n_stride+1)
                    jj = j*2048//(n_stride+1)
                    # print(i*2048//(n_stride+1), i*2048//(n_stride+1)+1024)
                    images_ = images[:, :, ii:ii+1024, jj:jj+1024]
                    outputs_ = model(images_)

                    # print(outputs_.shape)
                    outputs_ = torch.sigmoid(outputs_)
                    outputs[:, :, ii:ii+1024, jj:jj+1024] += outputs_
                    sum_num[ii:ii+1024, jj:jj+1024] += 1

            x = 829
            y = 1487
            # print('outputs3', outputs3[0, 19, y-1024:y+10-1024, x:x+10])
            # print('outputs before', outputs[0, 19, y:y+10, x:x+10])
            outputs = outputs/sum_num
            # print('outputs after', outputs[0, 19, y:y+10, x:x+10])
            
            outputs = (outputs > thr)
            outputs = outputs.detach().cpu().numpy()

            # exit()
            for output, image_name in zip(outputs, image_names):
                for c, segm in enumerate(output):
                    rle = encode_mask_to_rle(segm)
                    rles.append(rle)
                    filename_and_class.append(f"{IND2CLASS[c]}_{image_name}")
            
    return rles, filename_and_class

@hydra.main(version_base=None, config_path="configs", config_name="infer")
def main(cfg: DictConfig):
    L.seed_everything(cfg["seed"])

    x_min = 724
    y_min = 1200
    x_max = 1236
    y_max = 1712

    tf = A.Resize(resolution, resolution)
    tf_crop512 = A.Crop(x_min, y_min, x_max, y_max)
    
    test_dataset = XRayInferenceDataset(transforms=tf)
    test_dataset_crop512 = XRayInferenceDataset(transforms=tf_crop512)
    
    test_loader = DataLoader(
        dataset=test_dataset, 
        batch_size=8,
        shuffle=False,
        num_workers=4,
        drop_last=False
    )

    test_loader_crop512 = DataLoader(
        dataset=test_dataset_crop512, 
        batch_size=8,
        shuffle=False,
        num_workers=4,
        drop_last=False
    )
    
    model = instantiate(cfg["model"]["model"])
    model_subclass = instantiate(cfg["model_subclass"]["model"])

    new_weights = {}
    new_weights_subclass = {}
    weights = torch.load(PATH)['state_dict']   
    weights_subclass = torch.load(PATH_SUBCLASS)['state_dict']   

    for key in weights:
        new_weights[key[6:]] = weights[key]
    for key in weights_subclass:
        # print(key[6:])
        new_weights_subclass[key[6:]] = weights_subclass[key]

    model.load_state_dict(new_weights)
    model_subclass.load_state_dict(new_weights_subclass)

    rles, filename_and_class = ensemble_with_lessclassmodel_crop512(model, 
                                                                    model_subclass, 
                                                                    test_loader, 
                                                                    test_loader_crop512, 
                                                                    thr=0.5)
    # rles, filename_and_class = ensemble_with_patch_inOneImage(model, test_loader)

    classes, filename = zip(*[x.split("_") for x in filename_and_class])
    image_name = [os.path.basename(f) for f in filename]

    ## visualization 결과 저장-> 아직 안됨
    # for fc in filename_and_class :
    #     image = cv2.imread(os.path.join(IMAGE_ROOT, fc.split("_")[1]))
    #     preds = []
    #     for rle in rles[:len(CLASSES)]:
    #         pred = decode_rle_to_mask(rle, height=2048, width=2048)
    #         preds.append(pred)

    #     preds = np.stack(preds, 0)

    #     fig, ax = plt.subplots(1, 2, figsize=(24, 12))
    #     ax[0].imshow(image)    # remove channel dimension
    #     ax[1].imshow(label2rgb(preds))

    #     os.makedirs(os.path.join('vis_result',fc.split("_")[1].split('/')[0]), exist_ok=True)
    #     plt.savefig(os.path.join('vis_result',fc.split("_")[1]))

    df = pd.DataFrame({
        "image_name": image_name,
        "class": classes,
        "rle": rles,
    })

    df.to_csv(RESULT, index=False)
    print(RESULT, 'is saved')

if __name__ == "__main__":
    main()
