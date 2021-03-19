import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class BackBone(nn.Module):

    def __init__(self, base_model, input_size):
        super(BackBone, self).__init__()
        self.backbone = torch.nn.Sequential(*list(base_model.children())[:-1])
        dummy_input = torch.Tensor(np.random.rand(1, 3, input_size, input_size))
        dummy_feats = self.backbone(dummy_input)
        self.n_feats = dummy_feats.view(1, -1).shape[1]

    def forward(self, x):
        return self.backbone(x)


class CombinedModel(nn.Module):

    def __init__(self, base_model, input_size):
        super(CombinedModel, self).__init__()
        self.feature_extractor = BackBone(base_model=base_model, input_size=input_size)
        self.cls_pred = nn.Linear(self.feature_extractor.n_feats, 1)
        self.energy_pred = nn.Linear(self.feature_extractor.n_feats, 1)

    def forward(self, x):
        feats = self.feature_extractor(x)
        feats = feats.view(x.shape[0], -1)
        cls = torch.sigmoid(self.cls_pred(feats))
        energy = torch.exp(self.energy_pred(feats))
        return cls, energy


class SeparateModel(nn.Module):

    def __init__(self, base_model, input_size):
        super(SeparateModel, self).__init__()
        self.cls_fe = BackBone(base_model=base_model, input_size=input_size)
        self.reg_fe = BackBone(base_model=base_model, input_size=input_size)
        self.cls_pred = nn.Linear(self.cls_fe.n_feats, 1)
        self.reg_pred = nn.Linear(self.reg_fe.n_feats, 1)

    def compute_snr(self, x):
        return x

    def forward(self, x):
        cls_feats = self.cls_fe(x)
        reg_feats = self.reg_fe(x)
        cls_feats = cls_feats.view(x.shape[0], -1)
        reg_feats = reg_feats.view(x.shape[0], -1)
        # reg_feats = torch.cat([reg_feats, self.compute_snr(x)])
        cls = torch.sigmoid(self.cls_pred(cls_feats))
        reg = F.relu(self.reg_pred(reg_feats))
        return cls, reg


class SeparateModelSNR(nn.Module):

    def __init__(self, base_model, input_size):
        super(SeparateModelSNR, self).__init__()
        self.cls_fe = BackBone(base_model=base_model, input_size=input_size)
        self.reg_fe = BackBone(base_model=base_model, input_size=input_size)
        self.cls_pred = nn.Linear(self.cls_fe.n_feats, 1)
        self.reg_pred = nn.Linear(self.reg_fe.n_feats + 128, 1)

    def forward(self, image, snr):
        cls_feats = self.cls_fe(image)
        reg_feats = self.reg_fe(image)
        cls_feats = cls_feats.view(image.shape[0], -1)
        reg_feats = reg_feats.view(image.shape[0], -1)
        reg_feats = torch.cat([reg_feats, snr], 1)
        cls = torch.sigmoid(self.cls_pred(cls_feats))
        reg = F.relu(self.reg_pred(reg_feats))
        return cls, reg
