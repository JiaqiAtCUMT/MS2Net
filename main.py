import imageio
from torch.nn.parallel import DistributedDataParallel
from utils.summaries import TensorboardSummary
from model.networks import DualUNet_CWF_MSA, UNet, DualUNet_CWF_MSA_Number
import torch
import numpy as np
from loss_functions.Criteria import CrossEntropyLoss2d
import torch.backends.cudnn as cudnn
import utils.Transforms as myTransforms
from argparse import ArgumentParser
import torch.optim.lr_scheduler
from dataloader.dataset import Dataset_US3D, Dataset_ISPRS_Vaihingen, Dataset_ISPRS_Potsdam
import loss_functions.lovasz_loss as L
from utils.evaluator import Evaluator
import warnings
warnings.filterwarnings('ignore')
import os
from tqdm import tqdm
from apex import amp
from utils.saver import Saver

def val(args, val_loader, model, epoch):
    model.eval()
    epoch_loss = []
    val_loss = 0.0
    tbar = tqdm(val_loader, ncols=80)
    for i, (image, depth, target) in enumerate(tbar):
        image = image.transpose(1, 3)
        image = image.transpose(2, 3)
        image = image.type(torch.FloatTensor)

        depth = depth.unsqueeze(3)
        depth = depth.transpose(1, 3)
        depth = depth.transpose(2, 3)
        depth = depth.type(torch.FloatTensor)
        if args.onGPU == True:
            image = image.cuda()
            depth = depth.cuda()
            target = target.cuda()

        image_var = torch.autograd.Variable(image)
        depth_var = torch.autograd.Variable(depth)
        target_var = torch.autograd.Variable(target)

        # run the mdoel
        output = model(image_var, depth_var)
        # compute the loss
        loss = L.lovasz_softmax(output, target_var)
        epoch_loss.append(loss.item())

        # compute the confusion matrix
        pred = output.data.cpu().numpy()
        target = target_var.data.cpu().numpy()
        pred = np.argmax(pred, axis=1)  # 0-4
        evaluator.add_batch(target, pred)

        # print('[%d/%d] loss: %.3f time: %.2f' % (i, total_batches, loss.item(), time_taken))
        val_loss += loss.item()
        tbar.set_description('Train loss: %.3f' % (val_loss / (i + 1)))

    summary.visualize_image(writer, target, pred, epoch)
    average_epoch_loss_val = sum(epoch_loss) / len(epoch_loss)
    Acc = evaluator.Pixel_Accuracy()
    Acc_class = evaluator.Pixel_Accuracy_Class()
    mIoU = evaluator.Mean_Intersection_over_Union()
    FWIoU = evaluator.Frequency_Weighted_Intersection_over_Union()
    writer.add_scalar('valid/loss_epoch', average_epoch_loss_val, epoch)
    writer.add_scalar('valid/Acc', Acc, epoch)
    writer.add_scalar('valid/Acc_class', Acc_class, epoch)
    writer.add_scalar('valid/mIoU', mIoU, epoch)
    writer.add_scalar('valid/FWIoU', FWIoU, epoch)
    print(
        'Epoch [%d/%d], Loss: %.4f, [Validing] Acc: %.4f, Acc_class: %.4f, mIoU: %.4f, fwIoU: %.4f ' %
        (epoch, args.max_epochs, average_epoch_loss_val, Acc, Acc_class, mIoU, FWIoU))

    return average_epoch_loss_val, Acc, Acc_class, mIoU, FWIoU


def train(args, train_loader, model, optimizer, epoch):

    # switch to train mode
    model.train()
    train_loss = 0.0
    epoch_loss = []
    tbar = tqdm(train_loader, ncols=80)
    total_batches = len(train_loader)
    for i, (image, depth, target) in enumerate(tbar):
        image = image.transpose(1, 3)
        image = image.transpose(2, 3)
        image = image.type(torch.FloatTensor)

        depth = depth.unsqueeze(3)
        depth = depth.transpose(1, 3)
        depth = depth.transpose(2, 3)
        depth = depth.type(torch.FloatTensor)
        if args.onGPU == True:
            image = image.cuda()
            depth = depth.cuda()
            target = target.cuda()

        image_var = torch.autograd.Variable(image)
        depth_var = torch.autograd.Variable(depth)
        target_var = torch.autograd.Variable(target)

        # run the mdoel
        output = model(image_var, depth_var)
        # set the grad to zero
        optimizer.zero_grad()
        loss = L.lovasz_softmax(output, target_var)
        loss.backward()
        optimizer.step()

        epoch_loss.append(loss.item())
        writer.add_scalar('train/loss_iter', loss.item(), i+epoch*total_batches)

        pred = output.data.cpu().numpy()
        target = target_var.data.cpu().numpy()
        pred = np.argmax(pred, axis=1)  # 0-4
        evaluator.add_batch(target, pred)

        train_loss += loss.item()
        tbar.set_description('Train loss: %.3f' % (train_loss / (i + 1)))
    average_epoch_loss_train = sum(epoch_loss) / len(epoch_loss)
    writer.add_scalar('train/loss_epoch', average_epoch_loss_train, epoch)

    Acc = evaluator.Pixel_Accuracy()
    Acc_class = evaluator.Pixel_Accuracy_Class()
    mIoU = evaluator.Mean_Intersection_over_Union()
    FWIoU = evaluator.Frequency_Weighted_Intersection_over_Union()
    print(
        'Epoch [%d/%d], Loss: %.4f, [Training] Acc: %.4f, Acc_class: %.4f, mIoU: %.4f, fwIoU: %.4f ' %
        (epoch, args.max_epochs, average_epoch_loss_train, Acc, Acc_class, mIoU, FWIoU))
    return average_epoch_loss_train, Acc, Acc_class, mIoU, FWIoU

def netParams(model):
    total_paramters = 0
    for parameter in model.parameters():
        i = len(parameter.size())
        p = 1
        for j in range(i):
            p *= parameter.size(j)
        total_paramters += p

    return total_paramters


def trainValidateSegmentation(args):
    '''
    Main function for trainign and validation
    :param args: global arguments
    :return: None
    '''

    model = DualUNet_CWF_MSA(in_ch=3, d_ch=1, out_ch=args.classes, mode=args.mode)
    args.savedir = args.savedir + '/' + args.dataset + '/'

    if args.onGPU:
        model = model.cuda()
    # create the directory if not exist
    if not os.path.exists(args.savedir):
        os.makedirs(args.savedir)

    total_paramters = netParams(model)
    print('Total network parameters: ' + str(total_paramters))

    criteria = CrossEntropyLoss2d(weight=None)  # weight

    if args.onGPU:
        criteria = criteria.cuda()

    if args.dataset == 'US3D':
        train_loader = torch.utils.data.DataLoader(
            Dataset_US3D(train=True),
            batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

        val_loader = torch.utils.data.DataLoader(
            Dataset_US3D(train=False),
            batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    if args.dataset == 'ISPRS_Vaihingen':
        train_loader = torch.utils.data.DataLoader(
            Dataset_ISPRS_Vaihingen(transform=None, train=True),
            batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

        val_loader = torch.utils.data.DataLoader(
            Dataset_ISPRS_Vaihingen(transform=None, train=False),
            batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    if args.dataset == 'ISPRS_Potsdam':
        train_loader = torch.utils.data.DataLoader(
            Dataset_ISPRS_Potsdam(transform=None, train=True),
            batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

        val_loader = torch.utils.data.DataLoader(
            Dataset_ISPRS_Potsdam(transform=None, train=False),
            batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)


    if args.onGPU:
        cudnn.benchmark = True

    start_epoch = 0

    if args.resume is not None:
        if not os.path.exists(args.resume):
            print("Can not find resume path, please check again!")
        else:
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch']
            args.lr = checkpoint['lr']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))

    logFileLoc = args.savedir + args.logFile
    if os.path.isfile(logFileLoc):
        logger = open(logFileLoc, 'a')
    else:
        logger = open(logFileLoc, 'w')
        logger.write("Parameters: %s" % (str(total_paramters)))
        logger.write("\n%s\t%s\t%s\t%s\t%s\t" % ('Epoch', 'Loss(Tr)', 'Loss(val)', 'mIOU (tr)', 'mIOU (val'))
    logger.flush()

    optimizer = torch.optim.Adam(model.parameters(), args.lr, (0.9, 0.999), eps=1e-08, weight_decay=5e-4)
    # optimizer = torch.optim.SGD(model.parameters(), args.lr, 0.9, weight_decay=5e-8)
    model = torch.nn.DataParallel(model)
    # we step the loss by 2 after step size is reached
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_loss, gamma=0.5)
    for epoch in range(start_epoch, args.max_epochs):
        scheduler.step(epoch)
        lr = 0
        for param_group in optimizer.param_groups:
            lr = param_group['lr']
        print("Learning rate: " + str(lr))

        evaluator.reset()
        lossTr, Acc_tr, Acc_class_tr, mIoU_tr, FWIoU_tr = train(args, train_loader, model, optimizer, epoch)
        evaluator.reset()
        with torch.no_grad():
            lossVal, Acc_val, Acc_class_val, mIoU_val, FWIoU_val = val(args, val_loader, model, epoch)

        if mIoU_val > saver.best_mIoU:
            is_best = True
        else:
            is_best = False
        saver.save_checkpoint(epoch, model, optimizer, lossTr, lossVal, mIoU_tr, mIoU_val, lr, Acc_tr, Acc_val, Acc_class_tr, Acc_class_val, FWIoU_tr, FWIoU_val, is_best)

        if args.save_model_epoch != 0 and ((epoch+1) % args.save_model_epoch==0):
            saver.save_model_epoch(epoch, model)

        logger.write("\n%d\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.7f" % (epoch, lossTr, lossVal, mIoU_tr, mIoU_val, lr))
        logger.flush()
    logger.close()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--max_epochs', type=int, default=150, help='Max. number of epochs')
    parser.add_argument('--num_workers', type=int, default=4, help='No. of parallel threads')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size.')
    parser.add_argument('--step_loss', type=int, default=100, help='Decrease learning rate after how many epochs.')
    parser.add_argument('--lr', type=float, default=1e-4, help='Initial learning rate')
    parser.add_argument('--savedir', default='/root/Desktop/code_rewrite/MS2Net/train_result', help='directory to save the results')
    parser.add_argument('--resume', type=str, default=None,
                        help='Use this flag to load last checkpoint for training')  
    parser.add_argument('--classes', type=int, default=6, help='No of classes in the dataset. 5 for dfs and 6 for class+background')
    parser.add_argument('--logFile', default='trainValLog.txt',
                        help='File that stores the training and validation logs')
    parser.add_argument('--onGPU', default=True, help='Run on CPU or GPU. If TRUE, then GPU.')
    parser.add_argument('--dataset', default="ISPRS_Vaihingen", help='US3D or ISPRS_Potsdam or ISPRS_Vaihingen.')
    parser.add_argument('--mode', default="ToRGB", help="Bi-direction|ToRGB|ToDepth")
    parser.add_argument('--number', default=5)
    parser.add_argument('--save_model_epoch', type=int, default=0, help="Save parameters of the model every * epoches.")


    args = parser.parse_args()
    evaluator = Evaluator(args.classes)
    summary = TensorboardSummary(args)
    writer = summary.create_summary()
    saver = Saver(args, 0)
    trainValidateSegmentation(args)