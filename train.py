import os
import re
import numpy as np
import cv2
import torch
import torchvision
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

from idao_dataset import IdaoDataset, FFT, IdaoDatasetSNR
from idao_models import CombinedModel, SeparateModel, SeparateModelSNR


train_root_dir = './data/train'
IMAGE_SIZE = 576
RESIZE_H = 64
BATCH_SIZE = 4
INITIAL_LR = 2e-4
CONF_THRESH = 0.5
NUM_EPOCHS = 100
CLS_LOSS_WEIGHT = 1.0
REG_LOSS_WEIGHT = 1.0
MODEL_NAME = 'separate_r50'
device = 'cpu'


transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Resize(RESIZE_H),
    # torchvision.transforms.Grayscale(num_output_channels=1),
    # FFT()
])

ds = IdaoDatasetSNR(root=train_root_dir, transform=transforms)
n_train = int(0.95 * len(ds))
train_ds, valid_ds = torch.utils.data.random_split(ds, lengths=[n_train, len(ds) - n_train])

train_loader = torch.utils.data.DataLoader(train_ds, shuffle=True, batch_size=BATCH_SIZE, num_workers=2, drop_last=True)
valid_loader = torch.utils.data.DataLoader(valid_ds, shuffle=True, batch_size=BATCH_SIZE, drop_last=True)

model = SeparateModelSNR(torchvision.models.resnet50(pretrained=True), RESIZE_H).to(device)
loss_fn_cls = nn.BCELoss()
loss_fn_reg = nn.L1Loss()
optimizer_cls = torch.optim.Adam
optimizer = optimizer_cls(model.parameters(), lr=INITIAL_LR)

for epoch in range(NUM_EPOCHS):
    print("Epoch %d" % epoch)
    model.train()
    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    for i, (images, (cls, energy), snr) in pbar:
        images = images.to(device)
        cls = cls.unsqueeze(1).float().to(device)
        energy = energy.unsqueeze(1).to(device)
        snr = snr.to(device)
        cls_pred, reg_pred = model(images, snr)

        optimizer.zero_grad()
        loss_reg = loss_fn_reg(reg_pred, energy)
        loss_cls = loss_fn_cls(cls_pred, cls)
        loss = REG_LOSS_WEIGHT * loss_reg + CLS_LOSS_WEIGHT * loss_cls
        loss.backward()
        optimizer.step()
        if i % 25 == 0:
            pbar.set_description(
                f'loss reg: {loss_reg.detach().cpu().numpy():.4f} loss cls: {loss_cls.detach().cpu().numpy():.4f}')
    pbar = tqdm(enumerate(valid_loader), total=len(valid_loader))
    model.eval()
    cls_preds, reg_preds, cls_gt, reg_gt = [], [], [], []
    with torch.no_grad():
        for i, (images, (cls, energy)) in pbar:
            images = images.cuda()
            cls = cls.unsqueeze(1).float().cuda()
            energy = energy.unsqueeze(1).float().cuda()
            cls_pred, reg_pred = model(images)
            cls_pred = (cls_pred > CONF_THRESH).float()
            cls_preds.append(cls_pred.cpu().numpy())
            reg_preds.append(reg_pred.cpu().numpy())
            cls_gt.append(cls.cpu().numpy())
            reg_gt.append(energy.cpu().numpy())
    torch.save(model.state_dict(), f'./data/{MODEL_NAME}.pth')
    cls_gt = np.vstack(cls_gt)
    reg_gt = np.vstack(reg_gt)
    cls_preds = np.vstack(cls_preds)
    reg_preds = np.vstack(reg_preds)
    auc_score = roc_auc_score(cls_gt, cls_preds)
    l1_score = np.abs(reg_gt - reg_preds).mean()
    print(f'AUC: {auc_score:.4f} L1: {l1_score:.4f} IDAO: {(auc_score - l1_score) * 1000.0:.4f}')
