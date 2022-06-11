import numpy as np
import torch

def get_rgb(dataset=None):

    if dataset=='US3D':
        return np.array([
            [255, 0, 0],
            [204, 255, 0],
            [0, 255, 102],
            [0, 102, 255],
            [204, 0, 255],
            [255, 255, 255]])

    if dataset=='NYUv2':
        return np.array(
            [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128], [128, 0, 128], [0, 128, 128],
             [128, 128, 128],
             [64, 0, 0],
             [192, 0, 0], [64, 128, 0], [192, 128, 0], [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
             [0, 64, 0], [128, 64, 0],
             [0, 192, 0], [128, 192, 0], [0, 64, 128], [128, 64, 128], [0, 192, 128], [128, 192, 128], [64, 64, 0],
             [192, 64, 0],
             [64, 192, 0], [192, 192, 0], [64, 64, 128], [192, 64, 128], [64, 192, 128], [192, 192, 128], [0, 0, 64],
             [128, 0, 64],
             [0, 128, 64], [128, 128, 64], [0, 0, 192], [128, 0, 192], [0, 128, 192], [128, 128, 192], [64, 0, 64]])

    if dataset == 'ISPRS_Vaihingen' or 'ISPRS_Potsdam':
        return (np.array([
            [0, 153, 0],
            [198, 176, 68],
            [251, 255, 19],
            [182, 255, 5],
            [39, 255, 135],
            [194, 79, 68],
            [165, 165, 165],
            [105, 255, 248],
            [249, 255, 164],
            [28, 13, 255]
        ]))

def decode_segmap(label_mask, dataset, classes=0):
    if classes==0:
        raise Exception("The classes are illegal!")
    img_height = label_mask.shape[0]
    img_width = label_mask.shape[1]
    label_colours = get_rgb(dataset)
    r = label_mask.copy()
    g = label_mask.copy()
    b = label_mask.copy()
    for ll in range(0, classes):
        r[label_mask == ll] = label_colours[ll, 0]
        g[label_mask == ll] = label_colours[ll, 1]
        b[label_mask == ll] = label_colours[ll, 2]
    rgb = np.zeros((img_height, img_width, 3))
    rgb[:, :, 0] = r
    rgb[:, :, 1] = g
    rgb[:, :, 2] = b
    return rgb.astype(np.uint8)

def decode_seg_map_sequence(label_masks, dataset=None, classes=0):
    rgb_masks = []
    for label_mask in label_masks:
        rgb_mask = decode_segmap(label_mask, dataset, classes=classes)
        rgb_masks.append(rgb_mask)
    rgb_masks = torch.from_numpy(np.array(rgb_masks).transpose([0, 3, 1, 2]))
    return rgb_masks