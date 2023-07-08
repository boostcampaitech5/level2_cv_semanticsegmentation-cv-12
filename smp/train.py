import os

import albumentations as A
import hydra
import lightning as L
import torch.nn as nn
import wandb
from hydra.utils import instantiate
from lightning import Trainer
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint, RichProgressBar
from lightning.pytorch.loggers import WandbLogger
from models.base_module import Module
from omegaconf import DictConfig
from sklearn.model_selection import GroupKFold
# from loss import *
from importlib import import_module

from data.data_module import DataModule, NewXRayDataset, preprocessing


@hydra.main(version_base=None, config_path="configs", config_name="train")
def main(cfg: DictConfig):
    L.seed_everything(cfg["seed"])
    os.makedirs(f"/opt/ml/directory/baseline/checkpoints/{cfg['exp_name']}", exist_ok=True)
        
    # pickle 파일 생성
    pngs, pkls = preprocessing(make=cfg["make_pickle"])

    groups = [os.path.dirname(fname) for fname in pngs]
    y = [0] * len(pngs)
    group_kfold = GroupKFold(n_splits=cfg["fold"])

    # 한 환자의 두 개의 데이터가 train, val 나뉘어 지지 않게 group_kfold로 train, val split 진행
    for fold_idx, (train_idx, valid_idx) in enumerate(group_kfold.split(pngs, y, groups)):
        if cfg['k-fold'] == fold_idx:
            print("="*10 + f" fold : {fold_idx} " + "="*10)
            
            # Dataset
            train_data = (pngs[train_idx], pkls[train_idx])
            valid_data = (pngs[valid_idx], pkls[valid_idx])

            # Augmentation
            transforms = A.Compose([instantiate(aug) for _, aug in cfg["augmentation"].items()])

            train_dataset = NewXRayDataset(train_data, train=True, transforms=transforms)
            valid_dataset = NewXRayDataset(valid_data, train=True, transforms=transforms)

            datamodule = DataModule(train_dataset, valid_dataset, cfg)

            # Model load
            model = instantiate(cfg["model"]["model"])

            # Loss
            criterion = getattr(import_module('loss'), cfg['loss'])

            module = Module(model, criterion, cfg)
            exp_name = cfg["exp_name"] if fold_idx == 0 else f"{cfg['exp_name']}-{fold_idx}"
            
            # Lightning의 WandbLogger 사용 (from lightning.pytorch.loggers import WandbLogger)
            logger = [WandbLogger(project="Semantic Segmentation", name=str(cfg["exp_name"]), entity='ganddddi_segmentation', config=cfg)]
            
            # Lightning의 callback 관련 모듈을 리스트로 묶어준다. - (from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint, RichProgressBar)
            callbacks = [
                RichProgressBar(),
                ModelCheckpoint(
                    f"./checkpoints/{exp_name}",
                    f"best",
                    monitor="Valid Dice",
                    mode="max",
                    save_last=True,
                    save_weights_only=False,
                ),
                LearningRateMonitor(logging_interval="epoch"),
                EarlyStopping(monitor="Valid Dice", patience=5, mode="max"),
            ]
            
            # Lighning의 Trainer 모델 사용 - (from lightning import Trainer)
            trainer = Trainer(max_epochs=cfg["epoch"], logger=logger, callbacks=callbacks, precision="16-mixed")
            trainer.fit(module, datamodule=datamodule, ckpt_path= None if cfg['resume'] == 'None' else cfg['resume'])
            # trainer.fit(module, datamodule=datamodule)
            break

    wandb.finish()


if __name__ == "__main__":
    main()


