#!/usr/bin/env python3

import argparse
import os.path

from datasets import (
    DEFAULT_TRANSFORM,
    HeadPose_AFLW2000, HeadPose_AFLW, HeadPose_BIWI,
    AnglesFilter
)
from model import Network

import numpy as np
from scipy.special import softmax
import torch
from torch.utils.data import DataLoader

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs):
        return iterable


PROTOCOLS = [
    dict(AFLW2000=(HeadPose_AFLW2000, {}), BIWI=(HeadPose_BIWI, {})),
    dict(AFLW_test=(HeadPose_AFLW, dict(type_="test"))),
    dict(BIWI_test=(HeadPose_BIWI, dict(type_="test")))
]


def calc_angles(prediction):
    prediction = softmax(prediction, axis=-1)
    shape = prediction.shape

    bin_centers = np.linspace(-98.5, 98.5, 198).reshape(1, -1)
    prediction = (prediction.reshape(-1, 198) * bin_centers).sum(axis=1)

    return prediction.reshape(shape[:-1]).T


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-p', '--protocol', type=int, required=True,
        choices=[1, 2, 3],
        help='Testing protocol to use'
    )

    parser.add_argument(
        '-a', '--arch', required=True,
        choices=[f'resnet{k}' for k in (18, 34, 50, 101, 152)],
        help='Architecture to use'
    )

    parser.add_argument(
        '-b', '--batch-size', type=int, default=64,
        help='Batch size for inference (default: 64)'
    )

    return parser.parse_args()


def main():
    params = parse_args()

    model = Network(params.arch, os.path.abspath(
        os.path.join(
            os.path.dirname(__file__), os.path.pardir,
            'models', f'ap{params.protocol}', f'{params.arch}.pth'
        )
    ))
    if torch.cuda.is_available():
        model = model.cuda()

    for (name, (cls, kwargs)) in PROTOCOLS[params.protocol - 1].items():
        loader = DataLoader(
            AnglesFilter(cls(transform=DEFAULT_TRANSFORM, **kwargs)),
            batch_size=params.batch_size,
            shuffle=False
        )

        results = []

        for item in tqdm(loader, desc=name):
            input_ = item['image']
            if torch.cuda.is_available():
                input_ = input_.cuda()

            with torch.no_grad():
                output = calc_angles(model(input_).detach().cpu().numpy())
                expected = item['angles'].numpy()
                results.append(np.abs(output - expected))

        (mpitch, myaw, mroll) = np.concatenate(results, axis=0).mean(axis=0)
        mae = np.mean((mpitch, myaw, mroll))

        print(f"""
Dataset: {name}
    MAE:       {mae:.4f}
    mae_pitch: {mpitch:.4f}
    mae_yaw:   {myaw:.4f}
    mae_roll:  {mroll:.4f}
""".strip())


if __name__ == '__main__':
    main()
