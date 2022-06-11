
import os
import torch
import imageio
import numpy as np
from glob import glob
import torch.utils.data

class Dataset_US3D(torch.utils.data.Dataset):
    """
    This class is for Urban Semantic 3D Dataset.
    """
    def __init__(self, transform=None, train=True):
        self.train = train
        image_root = "/root/dataset/US3D/train/RGB"
        self.imList_all = glob(os.path.join(image_root, '*'))
        len_list = len(self.imList_all)
        # inter_point = int(len_list * 0.9)
        self.transform = transform
        if self.train:
            self.imList = self.imList_all
        else:
            image_root = "/root/dataset/US3D/test/RGB"
            self.imList_all = glob(os.path.join(image_root, '*'))
            self.imList = self.imList_all

    def __len__(self):
        return len(self.imList)

    def __getitem__(self, idx):
        if self.train:
            img_name = self.imList[idx].split('/')[-1]

            label_name = img_name.replace('RGB', 'NCLS')
            label_name = label_name.replace('.png', '.tif')
            label_path = os.path.join("/root/dataset/US3D/train/NCLS", label_name)
            label = np.array(imageio.imread(label_path)).astype(np.float32)

            depth_name = img_name.replace('RGB', 'DEP')
            depth_name = depth_name.replace('.png', '.tif')
            depth_path = os.path.join("/root/dataset/US3D/train/Depth", depth_name)
            depth = np.array(imageio.imread(depth_path)).astype(np.float32)

            image = np.array(imageio.imread(self.imList[idx])).astype(np.float32)
            # if self.transform:
            #     [image, label] = self.transform(image, label)
            image = torch.from_numpy(image)
            label = torch.from_numpy(label)
            return (image, depth, label)

        else:
            img_name = self.imList[idx].split('/')[-1]

            label_name = img_name.replace('RGB', 'NCLS')
            label_name = label_name.replace('.png', '.tif')
            label_path = os.path.join("/root/dataset/US3D/test/NCLS", label_name)
            label = np.array(imageio.imread(label_path)).astype(np.float32)

            depth_name = img_name.replace('RGB', 'DEP')
            depth_name = depth_name.replace('.png', '.tif')
            depth_path = os.path.join("/root/dataset/US3D/test/Depth", depth_name)
            depth = np.array(imageio.imread(depth_path)).astype(np.float32)

            image = np.array(imageio.imread(self.imList[idx])).astype(np.float32)
            # if self.transform:
            #     [image, label] = self.transform(image, label)
            image = torch.from_numpy(image)
            label = torch.from_numpy(label)
            return (image, depth, label)


class Dataset_ISPRS_Vaihingen(torch.utils.data.Dataset):
    """
    This class is for ISPRS Vaihingen Dataset.
    """
    def __init__(self, transform=None, train=True):
        self.train = train
        image_root = "/root/dataset/Vaihingen/TrainImage/rgb"
        self.imList_all = glob(os.path.join(image_root, '*'))
        len_list = len(self.imList_all)
        # inter_point = int(len_list * 0.9)
        self.transform = transform
        if self.train:
            self.imList = self.imList_all
        else:
            image_root = "/root/dataset/Vaihingen/TestImage/rgb"
            self.imList_all = glob(os.path.join(image_root, '*'))
            self.imList = self.imList_all

    def __len__(self):
        return len(self.imList)

    def __getitem__(self, idx):
        if self.train:
            img_name = self.imList[idx].split('/')[-1]
            label_path = os.path.join("/root/dataset/Vaihingen/TrainImage/labels", img_name)
            label = np.array(imageio.imread(label_path)).astype(np.float32)

            # depth_name = img_name.replace('.tif', '.jpg')
            depth_name = img_name
            depth_path = os.path.join("/root/dataset/Vaihingen/TrainImage/DSM", depth_name)
            depth = np.array(imageio.imread(depth_path)).astype(np.float32)

            image = np.array(imageio.imread(self.imList[idx])).astype(np.float32)
            # if self.transform:
            #     [image, label] = self.transform(image, label)
            image = torch.from_numpy(image)
            label = torch.from_numpy(label)
            return (image, depth, label)

        else:
            img_name = self.imList[idx].split('/')[-1]
            label_path = os.path.join("/root/dataset/Vaihingen/TestImage/labels", img_name)
            label = np.array(imageio.imread(label_path)).astype(np.float32)

            depth_name = img_name
            depth_path = os.path.join("/root/dataset/Vaihingen/TestImage/DSM", depth_name)
            depth = np.array(imageio.imread(depth_path)).astype(np.float32)

            image = np.array(imageio.imread(self.imList[idx])).astype(np.float32)
            # if self.transform:
            #     [image, label] = self.transform(image, label)
            image = torch.from_numpy(image)
            label = torch.from_numpy(label)
            return (image, depth, label)

class Dataset_ISPRS_Potsdam(torch.utils.data.Dataset):
    """
    This class is for ISPRS Potsdam Dataset.
    """
    def __init__(self, transform=None, train=True):
        self.train = train
        image_root = "/root/dataset/Potsdam/TrainImage/rgb"
        self.imList_all = glob(os.path.join(image_root, '*'))
        len_list = len(self.imList_all)
        # inter_point = int(len_list * 0.9)
        self.transform = transform
        if self.train:
            self.imList = self.imList_all
        else:
            image_root = "/root/dataset/Potsdam/TestImage/rgb"
            self.imList_all = glob(os.path.join(image_root, '*'))
            self.imList = self.imList_all

    def __len__(self):
        return len(self.imList)

    def __getitem__(self, idx):
        if self.train:
            img_name = self.imList[idx].split('/')[-1]
            label_path = os.path.join("/root/dataset/Potsdam/TrainImage/labels", img_name)
            label = np.array(imageio.imread(label_path)).astype(np.float32)

            # depth_name = img_name.replace('.tif', '.jpg')
            depth_name = 'dsm_' + img_name
            depth_path = os.path.join("/root/dataset/Potsdam/TrainImage/dsm", depth_name)
            depth = np.array(imageio.imread(depth_path)).astype(np.float32)

            image = np.array(imageio.imread(self.imList[idx])).astype(np.float32)
            # if self.transform:
            #     [image, label] = self.transform(image, label)
            image = torch.from_numpy(image)
            label = torch.from_numpy(label)
            return (image, depth, label)

        else:
            img_name = self.imList[idx].split('/')[-1]
            label_path = os.path.join("/root/dataset/Potsdam/TestImage/labels", img_name)
            label = np.array(imageio.imread(label_path)).astype(np.float32)

            depth_name = 'dsm_' + img_name
            depth_path = os.path.join("/root/dataset/Potsdam/TestImage/dsm", depth_name)
            depth = np.array(imageio.imread(depth_path)).astype(np.float32)

            image = np.array(imageio.imread(self.imList[idx])).astype(np.float32)
            # if self.transform:
            #     [image, label] = self.transform(image, label)
            image = torch.from_numpy(image)
            label = torch.from_numpy(label)
            return (image, depth, label)
