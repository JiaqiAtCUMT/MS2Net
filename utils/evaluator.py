import imageio
import numpy as np
from glob import glob
import os
import warnings
import cv2
from tqdm import tqdm
warnings.filterwarnings('ignore')

class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,)*2)

    def Pixel_Accuracy(self):
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return Acc

    def Pixel_Accuracy_Class(self):
        Acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        Acc = np.nanmean(Acc)
        return Acc

    def Mean_Intersection_over_Union(self):
        MIoU = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))
        MIoU = np.nanmean(MIoU)
        return MIoU

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def Kappa(self):
        p0 = self.Pixel_Accuracy()
        pe = np.dot(np.sum(self.confusion_matrix, axis=0), np.sum(self.confusion_matrix, axis=1)) / (self.confusion_matrix.sum() * self.confusion_matrix.sum())
        kappa = (p0 - pe) /(1 - pe)
        return kappa

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)

    def get_confusion_matrix(self):
        return self.confusion_matrix

if __name__ =='__main__':
    img_root = '/data/model/ModelLearn/Image_Segmentation/result/U_Net/IGRSS'
    label_root = '/data/dataset/IGRSS2020high_986valid_norm/crop_img_128/test/dfc/'
    img_path_list = sorted(glob((os.path.join(img_root, '*.tif'))))
    label_path_list = sorted(glob((os.path.join(label_root, '*.tif'))))[1900:]
    print('Found {} test images'.format(len(img_path_list)))
    evaluator = Evaluator(10)
    for i in tqdm(range(len(img_path_list))):
        img_gray = imageio.imread(img_path_list[i])
        label_gray = imageio.imread(label_path_list[i]) - 1
        evaluator.add_batch(label_gray, img_gray)
    print('Acc: {}, Acc_class: {}, MIoU: {}, fwIoU: {}'.format(evaluator.Pixel_Accuracy(), evaluator.Pixel_Accuracy_Class(),
                                                               evaluator.Mean_Intersection_over_Union(), evaluator.Frequency_Weighted_Intersection_over_Union()))

