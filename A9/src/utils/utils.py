import datetime
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from src.models import CNN_Model,ResNet18


class Utils:

    # helper function to un-normalize and display an image
    @staticmethod
    def imshow(img):
        img = img / 2 + 0.5  # unnormalize
        plt.imshow(np.transpose(img, (1, 2, 0)))  # convert from Tensor image

    def printdatetime(self):
        print("Model execution started at:" + datetime.datetime.today().ctime())

    # def printgpuinfo():
    # gpu_info = !nvidia-smi
    # gpu_info = '\n'.join(gpu_info)
    # if gpu_info.find('failed') >= 0:
    # 	print('Select the Runtime → "Change runtime type" menu to enable a GPU accelerator, ')
    # 	print('and then re-execute this cell.')
    # else:
    # 	print(gpu_info)

    def savemodel(model, epoch, path, optimizer_state_dict=None, train_losses=None, train_acc=None, test_acc=None,
                  test_losses=None):
        # Prepare model model saving directory.
        # save_dir = os.path.join(os.getcwd(), 'saved_models')
        t = datetime.datetime.today()

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer_state_dict,
            'train_losses': train_losses,
            'train_acc': train_acc,
            'test_losses': test_losses,
            'test_acc': test_acc,
            # 'lr_data': lr_data,
            # 'reg_loss_l1': reg_loss_l1
        }, path)

    def loadmodel(path):
        checkpoint = torch.load(path)
        # epoch = checkpoint['epoch']
        # model_state_dict = checkpoint['model_state_dict']
        # optimizer_state_dict = checkpoint['optimizer_state_dict']
        # train_losses = checkpoint['train_losses']
        # train_acc = checkpoint['train_acc']
        # test_losses = checkpoint['test_losses']
        # test_acc = checkpoint['test_acc']
        # lr_data = checkpoint['lr_data']
        return checkpoint

    def createmodel(checkpoint=None):
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        print(device)
        model = CNN_Model().to(device)

        return model, device

    def createmodelresnet18(checkpoint=None):
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        print(device)
        model = ResNet18().to(device)

        return model, device

    def createoptimizer(model, lr=0.1, momentum=0.9, weight_decay=0):
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

        return optimizer

    def createscheduler(optimizer, mode, factor, patience=5, verbose=True, threshold=0.01,
                        threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08):
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=mode, factor=factor, patience=patience,
                                                               verbose=verbose, threshold=threshold,
                                                               threshold_mode=threshold_mode,
                                                               cooldown=cooldown, min_lr=min_lr, eps=eps)

        return scheduler
