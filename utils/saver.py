import os
import torch
import shutil

class Saver(object):
    def __init__(self, args, best_mIoU):
        self.args = args
        self.best_mIoU = best_mIoU
        self.directory = os.path.join(self.args.savedir, args.dataset)


    def save_checkpoint(self, epoch, model, optimizer, lossTr, lossVal, mIoU_tr, mIoU_val, lr, Acc_tr, Acc_val, Acc_class_tr, Acc_class_val, FWIoU_tr, FWIoU_val, is_best):
        torch.save({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lossTr': lossTr,
            'lossVal': lossVal,
            'iouTr': mIoU_tr,
            'iouVal': mIoU_val,
            'lr': lr
        }, os.path.join(self.directory, 'checkpoint.pth'))

        if is_best:
            self.best_mIoU = mIoU_val
            model_file_name = self.directory + '/model_best' + '.pth'
            torch.save(model.state_dict(), model_file_name)
            with open(self.args.savedir + 'model_best.txt', 'w') as log:
                log.write(
                    "\nEpoch: %d\t Overall Acc (Tr): %.4f\t Overall Acc (Val): %.4f\t mIOU (Tr): %.4f\t mIOU (Val): %.4f" % (
                        epoch, Acc_tr, Acc_val, mIoU_tr, mIoU_val))
                log.write('\n')
                log.write('Per Class Training Acc: ' + str(Acc_class_tr))
                log.write('\n')
                log.write('Per Class Validation Acc: ' + str(Acc_class_val))
                log.write('\n')
                log.write('Training FWIOU: ' + str(FWIoU_tr))
                log.write('\n')
                log.write('Validation FWIOU: ' + str(FWIoU_val))


    def save_model_epoch(self, epoch, model):
        para_name = "model_" + str(epoch + 1) + ".pth"
        torch.save(model.state_dict(), os.path.join(self.directory, para_name))

