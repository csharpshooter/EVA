import datetime
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

from src.models import CNN_Model


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

    def savemodel(self, model, epoch, optimizer, train_losses, train_acc, test_acc, test_losses):
        # Prepare model model saving directory.
        # save_dir = os.path.join(os.getcwd(), 'saved_models')
        t = datetime.datetime.today()
        path = "/home/abhijit/Downloads/PytorchModels/EVA/A7/A7-" + str(t) + ".pth"
        print(t)
        print(path)

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_losses': train_losses,
            'train_acc': train_acc,
            'test_losses': test_losses,
            'test_acc': test_acc,
            # 'lr_data': lr_data,
            # 'reg_loss_l1': reg_loss_l1
        }, path)

    def loadmodel(path):
        checkpoint = torch.load(
            "/home/abhijit/Downloads/PytorchModels/EVA/A6/A6-None-2020-02-25 22:10:47.703768.pth")

        epoch = checkpoint['epoch']
        train_losses = checkpoint['train_losses']
        train_acc = checkpoint['train_acc']
        test_losses = checkpoint['test_losses']
        test_acc = checkpoint['test_acc']
        lr_data = checkpoint['lr_data']

    def createmodel(checkpoint=None):
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        print(device)
        model = CNN_Model().to(device)

        # if checkpoint != None:
        #     model.load_state_dict(checkpoint['model_state_dict'])

        return model, device

    def createoptimizer(model, lr=0.1, momentum=0.9, weight_decay=0):
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

        # if model != None:
        #     optimizer.load_state_dict(model['optimizer_state_dict'])

        return optimizer
