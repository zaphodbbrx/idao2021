import os
import re
import numpy as np
import torch
import torchvision


def parse_energy(img_name):
    return float(re.search('_[0-9]+_keV', img_name)[0].split('_')[1])


def parse_id(file_path):
    return os.path.splitext(os.path.basename(file_path))[0]


def signaltonoise(a, axis=0, ddof=0):
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return np.where(sd == 0, 0, m/sd)


class IdaoDataset(torchvision.datasets.ImageFolder):

    def __init__(self, train_mode=True, **kwargs):
        super(IdaoDataset, self).__init__(**kwargs)
        self.train_mode = train_mode

    def __getitem__(self, item):
        img, cls = super(IdaoDataset, self).__getitem__(item)
        img_path, _ = self.imgs[item]
        if self.train_mode:
            energy = parse_energy(img_path)
            return img, [float(cls), energy]
        else:
            id = parse_id(img_path)
            return img, id


class IdaoDatasetSNR(IdaoDataset):

    def __init__(self, **kwargs):
        super(IdaoDatasetSNR, self).__init__(**kwargs)

    def __getitem__(self, item):
        img, [cls, energy] = super(IdaoDatasetSNR, self).__getitem__(item)
        horizontal_snr = np.mean(signaltonoise(img, 1), 0)
        vertical_snr = np.mean(signaltonoise(img, 2), 0)
        snr_features = np.hstack([horizontal_snr, vertical_snr])
        return img, [cls, energy], snr_features


class FFT:

    def __call__(self, sample):
        fft = torch.fft.fft2(sample)
        res = torch.cat([sample, fft.real, fft.imag])
        return res
