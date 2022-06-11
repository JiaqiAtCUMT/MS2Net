import torch
import imageio
import torch.utils.data
from glob import glob
import os
import numpy as np

class DFSDataset_test(torch.utils.data.Dataset):
    def __init__(self, args, transform=None):

        self.dataset = args.dataset
        if self.dataset == 'US3D':
            image_root = "/private/Dataset/US3D/test_R_D_512/"
            self.imList = glob(os.path.join(image_root, '*RGB*'))

        if self.dataset == 'ISPRS':
            image_root = "/private/Dataset/ISPRS/ISPRS_Potsdam_512x/test/RGBIR"
            self.imList = glob(os.path.join(image_root, '*'))

    def __len__(self):
        return len(self.imList)

    def __getitem__(self, idx):
        if self.dataset == 'US3D':
            img_name = self.imList[idx].split('/')[-1]
            label_name = img_name.replace("RGB", "NCLS").replace(".png", ".tif")
            label_path = os.path.join("/private/Dataset/US3D/test_RAD_512/", label_name)
            label = np.array(imageio.imread(label_path)).astype(np.float32)

            depth_name = img_name.replace("RGB", "DEP").replace(".png", ".tif")
            depth_path = os.path.join("/private/Dataset/US3D/test_R_D_512/", depth_name)
            depth = np.array(imageio.imread(depth_path)).astype(np.float32)

            image = np.array(imageio.imread(self.imList[idx])).astype(np.float32)
            # if self.transform:
            #     [image, label] = self.transform(image, label)
            image = torch.from_numpy(image)
            label = torch.from_numpy(label)
            return (image, depth, label, label_name)

        if self.dataset == 'ISPRS':
            img_name = self.imList[idx].split('/')[-1]
            label_path = os.path.join("/private/Dataset/ISPRS/ISPRS_Potsdam_512x/test/Labels_background", img_name)
            label = np.array(imageio.imread(label_path)).astype(np.float32)

            depth_name = img_name.replace('.tif', '.jpg')
            depth_path = os.path.join("/private/Dataset/ISPRS/ISPRS_Potsdam_512x/test/DSM", depth_name)
            depth = np.array(imageio.imread(depth_path)).astype(np.float32)

            image = np.array(imageio.imread(self.imList[idx])).astype(np.float32)
            # if self.transform:
            #     [image, label] = self.transform(image, label)
            image = torch.from_numpy(image)
            label = torch.from_numpy(label)
            return_img_name = img_name.replace('.tif', '.png')
            return (image, depth, label, return_img_name)

