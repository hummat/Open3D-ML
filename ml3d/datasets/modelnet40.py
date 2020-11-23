from os.path import exists, join
from pathlib import Path
import logging
import h5py

import numpy as np

from .base_dataset import BaseDataset
from ..utils import make_dir, DATASET

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(asctime)s - %(module)s - %(message)s',
)
log = logging.getLogger(__name__)


class ModelNet40(BaseDataset):

    def __init__(self,
                 dataset_path,
                 name="ModelNet40",
                 cache_dir='./logs/cache',
                 use_cache=False,
                 class_weights=[
                     625, 106, 515, 173, 572, 335, 64, 197, 889, 167, 79, 137,
                     200, 109, 200, 149, 171, 155, 145, 124, 149, 284, 465, 200,
                     88, 231, 239, 104, 115, 128, 680, 124, 90, 392, 163, 344,
                     267, 475, 87, 103
                 ],
                 ignored_label_inds=[],
                 train_files=[
                     'ply_data_train0.h5', 'ply_data_train1.h5',
                     'ply_data_train2.h5', 'ply_data_train3.h5',
                     'ply_data_train4.h5'
                 ],
                 test_files=['ply_data_test0.h5', 'ply_data_test1.h5'],
                 test_result_folder='./test',
                 **kwargs):
        super().__init__(dataset_path=dataset_path,
                         name=name,
                         cache_dir=cache_dir,
                         use_cache=use_cache,
                         class_weights=class_weights,
                         ignored_label_inds=ignored_label_inds,
                         test_result_folder=test_result_folder,
                         **kwargs)

        self.label_to_names = self.get_label_to_names()
        self.num_classes = len(self.label_to_names)
        self.dataset_path = join(dataset_path, 'modelnet40_ply_hdf5_2048')

        self.train_files = train_files
        self.test_files = test_files

    @staticmethod
    def get_label_to_names():
        label_to_names = {
            0: 'airplane',
            1: 'bathtub',
            2: 'bed',
            3: 'bench',
            4: 'bookshelf',
            5: 'bottle',
            6: 'bowl',
            7: 'car',
            8: 'chair',
            9: 'cone',
            10: 'cup',
            11: 'curtain',
            12: 'desk',
            13: 'door',
            14: 'dresser',
            15: 'flower_pot',
            16: 'glass_box',
            17: 'guitar',
            18: 'keyboard',
            19: 'lamp',
            20: 'laptop',
            21: 'mantel',
            22: 'monitor',
            23: 'night_stand',
            24: 'person',
            25: 'piano',
            26: 'plant',
            27: 'radio',
            28: 'range_hood',
            29: 'sink',
            30: 'sofa',
            31: 'stairs',
            32: 'stool',
            33: 'table',
            34: 'tent',
            35: 'toilet',
            36: 'tv_stand',
            37: 'vase',
            38: 'wardrobe',
            39: 'xbox'
        }
        return label_to_names

    def get_split(self, split):
        return ModelNet40Split(self, split=split)

    def get_split_list(self, split):
        if split in ['test', 'testing']:
            files = self.test_files
        elif split in ['train', 'training', 'val', 'validation']:
            files = self.train_files
        elif split in ['all']:
            files = self.train_files + self.test_files
        else:
            raise ValueError(f"Invalid split {split}")
        return files

    def is_tested(self, attr):
        cfg = self.cfg
        name = attr['name']
        path = cfg.test_result_folder
        store_path = join(path, self.name, name + '.labels')
        if exists(store_path):
            print("{} already exists.".format(store_path))
            return True
        else:
            return False

    def save_test_result(self, results, attr):
        cfg = self.cfg
        name = attr['name']
        path = cfg.test_result_folder
        make_dir(path)

        pred = results['predict_labels'] + 1
        store_path = join(path, self.name, name + '.labels')
        make_dir(Path(store_path).parent)
        np.savetxt(store_path, pred.astype(np.int32), fmt='%d')

        log.info("Saved {} in {}.".format(name, store_path))


class ModelNet40Split:

    def __init__(self, dataset, split='training'):

        self.cfg = dataset.cfg
        self.path_list = dataset.get_split_list(split)
        self.split = split

        self.points = []
        self.labels = []
        for file in self.path_list:
            with h5py.File(join(dataset.dataset_path, file), 'r') as f:
                points = np.array(f['data'])
                labels = np.array(f['label'])
                self.points.append(points)
                self.labels.append(labels)

        self.points = np.concatenate(self.points).astype(np.float32)
        self.labels = np.concatenate(self.labels).astype(np.int64)

        # Train/validation split
        np.random.seed(42)
        rand_indx = np.random.permutation(len(self.labels))
        if split in ['train', 'training']:
            indx = rand_indx[len(rand_indx) // 10:]
        elif split in ['val', 'validation']:
            indx = rand_indx[:len(rand_indx) // 10]
        else:
            indx = rand_indx

        self.points = self.points[indx]
        self.labels = self.labels[indx]

    def __len__(self):
        return len(self.labels)

    def get_data(self, idx):
        return {'point': self.points[idx], 'label': self.labels[idx]}

    def get_attr(self, idx):
        name = f"{self.split}_{idx}"
        if idx < 2048:
            path = self.path_list[0]
        elif 2048 <= idx < 2048 * 2:
            path = self.path_list[1]
        elif 2048 <= idx < 2048 * 3:
            path = self.path_list[2]
        elif 2048 <= idx < 2048 * 4:
            path = self.path_list[3]
        else:
            path = self.path_list[4]
        return {'name': name, 'path': str(Path(path)), 'split': self.split}


DATASET._register_module(ModelNet40)
