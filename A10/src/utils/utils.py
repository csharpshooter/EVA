import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from albumentations.pytorch import ToTensor
from torchvision import datasets

import src.dataset.dataset as dst
from src.models import CNN_Model, ResNet18


class Utils:

    # helper function to un-normalize and display an image
    @staticmethod
    def imshow(img):
        img = img / 2 + 0.5  # unnormalize
        plt.imshow(np.transpose(img, (1, 2, 0)))
        # plt.imshow(np.transpose(img, (1, 2, 0)))  # convert from Tensor image

    def imshowt(tensor):
        tensor = tensor.squeeze()
        if len(tensor.shape) > 2: tensor = tensor.permute(1, 2, 0)
        img = tensor.cpu().numpy()
        plt.imshow(img, cmap='gray')
        plt.show()

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
                  test_losses=None, lr_data=None, class_correct=None, class_total=None):
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
            'lr_data': lr_data,
            'class_correct': class_correct,
            'class_total': class_total
        }, path)

    def loadmodel(path):
        checkpoint = torch.load(path)
        epoch = checkpoint['epoch']
        model_state_dict = checkpoint['model_state_dict']
        optimizer_state_dict = checkpoint['optimizer_state_dict']
        train_losses = checkpoint['train_losses']
        train_acc = checkpoint['train_acc']
        test_losses = checkpoint['test_losses']
        test_acc = checkpoint['test_acc']
        lr_data = checkpoint['lr_data']
        class_correct = checkpoint['class_correct']
        class_total = checkpoint['class_total']
        return checkpoint, epoch, model_state_dict, optimizer_state_dict, train_losses, train_acc, test_losses, test_acc \
            , test_losses, lr_data, class_correct, class_total

    def createmodel(model_state_dict=None):
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        print(device)
        model = CNN_Model().to(device)

        return model, device

    def createmodelresnet18(model_state_dict=None):
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        print(device)
        model = ResNet18().to(device)

        if model_state_dict != None:
            model.load_state_dict(state_dict=model_state_dict)

        return model, device

    def createoptimizer(model, lr=0.1, momentum=0.9, weight_decay=0, nesterov=False):
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay,
                              nesterov=nesterov)

        return optimizer

    def createscheduler(optimizer, mode, factor, patience=5, verbose=True, threshold=0.01,
                        threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08):
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=mode, factor=factor, patience=patience,
                                                               verbose=verbose, threshold=threshold,
                                                               threshold_mode=threshold_mode,
                                                               cooldown=cooldown, min_lr=min_lr, eps=eps)

        return scheduler

    def createschedulersteplr(optimizer, step_size=15, gamma=0.1):
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=step_size, gamma=gamma)

        return scheduler

    def calculatemeanandstddeviation(self=None):
        traindataset = datasets.CIFAR10('./data', train=True, download=True, transform=ToTensor())
        testdataset = datasets.CIFAR10('./data', train=False, download=True, transform=ToTensor())

        data = np.concatenate([traindataset.data, testdataset.data], axis=0)
        data = data.astype(np.float32) / 255.

        means = []
        stdevs = []

        for i in range(3):  # 3 channels
            pixels = data[:, :, :, i].ravel()
            means.append(np.mean(pixels))
            stdevs.append(np.std(pixels))

        return [means[0], means[1], means[2]], [stdevs[0], stdevs[1], stdevs[2]]

    def showaccuracyacrossclasses(class_correct, class_total):
        classes = dst.Dataset.getclassesinCIFAR10dataset()
        for i in range(10):
            if class_total[i] > 0:
                print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
                    classes[i], 100 * class_correct[i] / class_total[i],
                    np.sum(class_correct[i]), np.sum(class_total[i])))
            else:
                print('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))

        print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
            100. * np.sum(class_correct) / np.sum(class_total),
            np.sum(class_correct), np.sum(class_total)))
