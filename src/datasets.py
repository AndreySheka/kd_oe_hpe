from albumentations import Compose, Normalize, Resize
from albumentations.pytorch.transforms import ToTensorV2

from torch.utils.data.dataset import Dataset, IterableDataset

from scipy.io import loadmat
from scipy.spatial.transform import Rotation
import cv2
import numpy as np

import json
import os.path
import sqlite3


__all__ = [
    'HeadPose_300WLP',
    'HeadPose_AFLW',
    'HeadPose_AFLW2000',
    'HeadPose_BIWI',

    'DEFAULT_TRANSFORM',
    'AnglesFilter'
]


DEFAULT_TRANSFORM = Compose([Resize(224, 224), Normalize(), ToTensorV2()])


def load_mat(filename):
    mat = loadmat(filename)
    angles = np.degrees(mat['Pose_Para'][0][:3])
    return angles * [-1, 1, 1]


def load_pose(filename):
    with open(filename) as f:
        rotmat = []
        for line in f:
            line = list(map(float, line.strip().split()))
            if line:
                rotmat.append(line)
            else:
                break

    return Rotation.from_matrix(rotmat).as_euler('XYZ', degrees=True)


class BaseDataset(Dataset):
    ROOT = os.path.abspath(os.path.dirname(__file__) + '/..')
    FOLDER = ""
    JSON_FILE = ""

    def __init__(self, transform=None, type_="all"):
        self.data = []

        with open(os.path.join(self.ROOT, 'data', self.JSON_FILE)) as f:
            for (filename, infos) in json.load(f).items():
                for info in infos:
                    if type_ in ("all", info.get("type")):
                        self.data.append({
                            'file': filename,
                            **info
                        })

        self.transform = transform

    def get_angles(self, index):
        raise NotImplementedError

    def __getitem__(self, index):
        info = self.data[index]
        image = cv2.imread(os.path.join(
            self.ROOT, 'data', self.FOLDER, info['file']
        ))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        (x1, y1, x2, y2) = info['bbox']
        x1 = max(0, int(x1))
        y1 = max(0, int(y1))
        x2 = int(x2)
        y2 = int(y2)

        image = image[y1:y2, x1:x2]
        if self.transform is not None:
            image = self.transform(image=image)['image']

        return dict(
            image=image,
            angles=self.get_angles(index)
        )


class HeadPose_300WLP(BaseDataset):
    FOLDER = '300W_LP'
    JSON_FILE = '300W_LP.json'

    def get_angles(self, index):
        return load_mat(os.path.join(
            self.ROOT, 'data', self.FOLDER,
            os.path.splitext(self.data[index]['file'])[0] + '.mat'
        ))


class HeadPose_AFLW(BaseDataset):
    FOLDER = 'AFLW/data/flickr'
    JSON_FILE = 'AFLW.json'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.angles = {}

        with sqlite3.connect(os.path.join(
            self.ROOT, 'data', 'AFLW', 'aflw.sqlite'
        )) as db:
            cursor = db.cursor()
            cursor.execute("SELECT face_id, pitch, yaw, roll FROM FacePose")
            for (face_id, pitch, yaw, roll) in cursor.fetchall():
                self.angles[face_id] = np.degrees((-pitch, yaw, -roll))

    def get_angles(self, index):
        return self.angles[self.data[index]['faceid']]


class HeadPose_AFLW2000(BaseDataset):
    FOLDER = 'AFLW2000'
    JSON_FILE = 'AFLW2000.json'

    def get_angles(self, index):
        return load_mat(os.path.join(
            self.ROOT, 'data', self.FOLDER,
            os.path.splitext(self.data[index]['file'])[0] + '.mat'
        ))


class HeadPose_BIWI(BaseDataset):
    FOLDER = 'BIWI/hpdb'
    JSON_FILE = 'BIWI.json'

    def get_angles(self, index):
        return load_pose(os.path.join(
            self.ROOT, 'data', self.FOLDER,
            self.data[index]['file'].replace('_rgb.png', '_pose.txt')
        ))


class AnglesFilter(IterableDataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __iter__(self):
        return iter((
            item
            for item in self.dataset
            if (np.abs(item['angles']) <= 99).all()
        ))
