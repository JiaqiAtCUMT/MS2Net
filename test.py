import imageio
import numpy as np
import torch
import os
from dataloader.dfs_test_dataset import DFSDataset_test
from argparse import ArgumentParser
from utils.evaluator import Evaluator
from model.networks import DualUNet, DualUNet_CWF, DualUNet_MSA, DualUNet_CWF_MSA_Number, DualUNet_CWF_MSA
import tifffile
from tqdm import tqdm
evaluator_1 = Evaluator(5)

def get_rgb(args):
    if args.dataset == 'US3D':
        return np.array([
            [255, 0, 0],
            [204, 255, 0],
            [0, 255, 102],
            [0, 102, 255],
            [204, 0, 255],
            [255, 255, 255]])

    if args.dataset == 'ISPRS':
        return (np.array([
            [0, 255, 255],
            [0, 255, 0],
            [255, 255, 0],
            [0, 0, 255],
            [255, 255, 255],
            [255, 0, 0]
        ]))

def decode_segmap(label_mask, args):
    label_colours = get_rgb(args)
    if len(label_mask.shape) == 3:
        label_mask = np.squeeze(label_mask, axis=0)
    h, w = label_mask.shape[0], label_mask.shape[1]
    r = label_mask.copy()
    g = label_mask.copy()
    b = label_mask.copy()
    for ll in range(0, args.classes):
        r[label_mask == ll] = label_colours[ll, 0]
        g[label_mask == ll] = label_colours[ll, 1]
        b[label_mask == ll] = label_colours[ll, 2]
    rgb = np.zeros((h, w, 3))
    rgb[:, :, 0] = r
    rgb[:, :, 1] = g
    rgb[:, :, 2] = b
    return rgb.astype(np.uint8)


def main(args):
    test_loader = torch.utils.data.DataLoader(
        DFSDataset_test(args),
        batch_size=args.batch_size, shuffle=True, pin_memory=True)
    if args.model == 'DualUNet':
        model = DualUNet(3, 1, args.classes)
        args.weightDir = 'MS2Net\code\MS2Net_Projects\train_result\ablation_backbone\US3D_7953'
    if args.model == 'DualUNet_MSF':
        model = DualUNet_CWF(3, 1, args.classes)
        args.weightDir = 'MS2Net\code\MS2Net_Projects\train_result\ablation_MSF\US3D_8154'
    if args.model == 'DualUNet_MSA':
        model = DualUNet_MSA(3, 1, args.classes)
        args.weightDir = 'MS2Net\code\MS2Net_Projects\train_result\ablation_MSA\US3D_Dep2RGB_8147'
    if args.model == 'DualUNet_MSF_MSA':
        model = DualUNet_CWF_MSA(3, 1, args.classes, 'ToRGB')
        args.weightDir = 'MS2Net\code\MS2Net_Projects\train_result\ablation_stage\5-ToRGB\US3D'
    
    if args.model == 'Ablation_Number':
        model = DualUNet_CWF_MSA_Number(3, 1, args.classes, args.numbers)
        args.weightDir = 'MS2Net\code\MS2Net_Projects\train_result\ablation_stage' + str(args.numbers) + '-ToRGB\US3D'
         
    if args.model == 'Direction':
        model = DualUNet_CWF_MSA(3, 1, args.classes, args.directions)
        args.weightDir = 'MS2Net\code\MS2Net_Projects\train_result\ablation_direction\5-' + str(args.directions) + '-ToRGB\US3D'
    if args.gpu:
        model = model.cuda()
    model = torch.nn.DataParallel(model)

    weight_name = 'model_' + str(args.test_epoch) + '.pth'
    model_weight_file = os.path.join(args.root, args.weightDir, weight_name)
    if not os.path.isfile(model_weight_file):
        print('Pre-trained model file does not exist. Please check ../pretrained/decoder folder')
        exit(-1)
    else:
        print("Loading {}...".format(model_weight_file))
    model_param = torch.load(model_weight_file)
    model.load_state_dict(model_param)

    # set to evaluation mode
    model.eval()
    tbar = tqdm(test_loader)
    for i, (image, depth, target, image_name) in enumerate(tbar):
        # start_time = time.time()
        image = image.transpose(1, 3)
        image = image.transpose(2, 3)
        image = image.type(torch.FloatTensor)

        depth = depth.unsqueeze(3)
        depth = depth.transpose(1, 3)
        depth = depth.transpose(2, 3)
        depth = depth.type(torch.FloatTensor)
        # print(input.shape)
        if args.gpu == True:
            image = image.cuda()
            depth = depth.cuda()
            target = target.cuda()

        image_var = torch.autograd.Variable(image)
        depth_var = torch.autograd.Variable(depth)
        target_var = torch.autograd.Variable(target)

        # run the mdoel
        with torch.no_grad():
            output = model(image_var, depth_var)
        # compute the loss

        pred_1 = output.data.cpu().numpy()
        target = target_var.data.cpu().numpy()
        pred_1 = np.argmax(pred_1, axis=1)  # 0-4
        evaluator_1.add_batch(target, pred_1)


        if args.get_gray_output:
            gray_output_path = os.path.join(args.SaveDir, args.dataset, 'gray_pred')
            if not os.path.exists(gray_output_path):
                os.makedirs(gray_output_path)
            for i in range(pred_1.shape[0]):
                pred_name = image_name[i]
                tifffile.imwrite(os.path.join(gray_output_path, pred_name), pred_1[i])

        if args.get_rgb_output:
            rgb_output_path = os.path.join(args.SaveDir, args.dataset, 'rgb_pred')
            if not os.path.exists(rgb_output_path):
                os.makedirs(rgb_output_path)
            for i in range(pred_1.shape[0]):
                pred_name = image_name[i].replace(".tif", ".png")
                image = decode_segmap(pred_1[i], args)
                tifffile.imwrite(os.path.join(rgb_output_path, pred_name), image)

    Acc_1 = evaluator_1.Pixel_Accuracy()
    Acc_class_1 = evaluator_1.Pixel_Accuracy_Class()
    mIoU_1 = evaluator_1.Mean_Intersection_over_Union()
    FWIoU_1 = evaluator_1.Frequency_Weighted_Intersection_over_Union()
    kappa_1 = evaluator_1.Kappa()

    if args.get_confusion_matrix:
        confusion_matrix_path = os.path.join(args.SaveDir, args.dataset, 'confusion_matrix')
        if not os.path.exists(confusion_matrix_path):
            os.makedirs(confusion_matrix_path)
        confusion_matrix_1 = evaluator_1.get_confusion_matrix()
        imageio.imwrite(os.path.join(confusion_matrix_path, 'confusion_matrix_1.tif'), confusion_matrix_1)

    print(
        'Acc_1: %.4f, Acc_class_1: %.4f, mIoU_1: %.4f, FWIoU_1: %.4f, Kappa_1: %.4f ' %
        (Acc_1, Acc_class_1, mIoU_1, FWIoU_1, kappa_1))

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--gpu', default=True, type=bool, help='Run on CPU or GPU. If TRUE, then GPU.')
    parser.add_argument('--model', default='DualUNet', type=str, help='DualUNet, DualUNet_MSF, DualUNet_MSA')
    parser.add_argument('--root', default='./', type=str)
    parser.add_argument('--dataset', default='US3D', type=str)
    parser.add_argument('--numbers', default=4, type=int, help='1,2,3,4,5')
    parser.add_argument('--directions', default="ToRGB", type=str, help='ToRGB, ToDepth, Bi-direction')
    parser.add_argument('--batch_size', default=15, type=int)
    parser.add_argument('--weightDir', default=None)
    parser.add_argument('--classes', default=5, type=int, help='Number of classes in the dataset. 20 for Cityscapes')
    parser.add_argument('--test_epoch', default='best'
                        , type=str, help='Number of epochs to be chosen for test')
    parser.add_argument('--SaveDir', default='../output')
    parser.add_argument('--get_rgb_output', type=bool, default=False)
    parser.add_argument('--get_gray_output', type=bool, default=False)
    parser.add_argument('--get_confusion_matrix', type=bool, default=False)

    args = parser.parse_args()
    main(args)
