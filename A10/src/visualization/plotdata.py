import matplotlib.pyplot as plt
import numpy as np
import torch

from src.utils import utils
from src.visualization.gradcam import gradcamhelper


# from src.visualization.gradcam.gradcam import gradcamof


class PlotData:

    def showImagesfromdataset(dataiterator, classes):
        images, labels = dataiterator.next()
        images = images.numpy()  # convert images to numpy for display

        # plot the images in the batch, along with the corresponding labels
        fig = plt.figure(figsize=(25, 4))
        # display 20 images
        for idx in np.arange(20):
            ax = fig.add_subplot(2, 20 / 2, idx + 1, xticks=[], yticks=[])
            utils.Utils.imshow(images[idx])
            ax.set_title(classes[labels[idx]])

        plt.savefig("images/imagesfromdataset.png", bbox_inches='tight')

    def plotmisclassifiedimages(dataiterator, model, classes, batch_size, dogradcam=False, device=None):

        # plot the images in the batch, along with predicted and true labels
        fig = plt.figure(figsize=(20, 20))

        loc = 0
        count = 30
        while loc < count:

            img, labels = dataiterator.next()
            # images = img.numpy()

            # move model inputs to cuda
            images = img.cuda()

            # get sample outputs
            output = model(images)
            # convert output probabilities to predicted class
            _, preds_tensor = torch.max(output, 1)
            preds = np.squeeze(preds_tensor.cpu().numpy())

            for idx in np.arange(batch_size):
                if preds[idx] != labels[idx].item():
                    ax = fig.add_subplot(6, 5, loc + 1, xticks=[], yticks=[])

                    if dogradcam != True:
                        utils.Utils.imshow(images[idx].cpu())

                        ax.set_title("Pred={} (Act={})".format(classes[preds[idx]], classes[labels[idx]])
                                     , color="red")
                    else:

                        # utils.Utils.imshow(images[idx].cpu())

                        gradcamimage, prediction = gradcamhelper.dogradcam(image=images[idx].unsqueeze(0), model=model,
                                                                           device=device,
                                                                           classes=classes)
                        tensor = gradcamimage[0].squeeze()
                        tensor = tensor.permute(1, 2, 0)
                        img = tensor.cpu().numpy()
                        plt.imshow(img, cmap="gray", interpolation='nearest', aspect="auto")

                        ax.set_title("Pred={} (Act={})".format(classes[preds[idx]], classes[labels[idx]])
                                     , color="red")
                        loc += 1

                        if loc >= count:
                            break

            plt.savefig("images/missclassifiedimages.png", bbox_inches='tight')

    def plottesttraingraph(train_losses, train_acc, test_losses, test_acc, lr_data, epochs, plotonsamegraph=False,
                           doProcessArray=False):

        if doProcessArray == True:
            train_acc = utils.Utils.processarray(train_acc, epochs)
            train_losses = utils.Utils.processarray(train_losses, epochs)

        maxtestacc = round(max(test_acc), 2)
        maxtrainacc = round(max(train_acc), 2)
        # testloss = min(test_losses)
        # trainloss = min(train_losses)

        if plotonsamegraph == True:
            fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(25, 10))
            l, = axs[0].plot(train_losses, linestyle='--', label="Train Loss")
            l1, = axs[0].plot(test_losses, label="Test Loss")
            axs[0].set_title("Training and Test Loss.")
            axs[0].legend(loc="best", ncol=1, handles=[l, l1])
            t, = axs[1].plot(train_acc, linestyle='--', label="Train Accuracy")
            t1, = axs[1].plot(test_acc, label="Test Accuracy")
            axs[1].set_title(
                "Training and Test Accuracy. Max train acc = {}, max test acc = {}.".format(maxtrainacc, maxtestacc))
            axs[1].legend(loc="best", ncol=1, handles=[t, t1])
            axs[2].plot(lr_data)
            axs[2].set_title("Learning Rate")

        else:
            fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(15, 10))
            axs[0, 0].plot(train_losses)
            axs[0, 0].set_title("Training Loss")
            axs[1, 0].plot(train_acc)
            axs[1, 0].set_title("Training Accuracy")
            axs[0, 1].plot()
            axs[0, 1].set_title("Test Loss")
            axs[1, 1].plot()
            axs[1, 1].set_title("Test Accuracy")
            axs[2, 0].plot(lr_data)
            axs[2, 0].set_title("Learning Rate")

        plt.savefig("images/traintestgraphs.png", bbox_inches='tight')
        plt.plot()
        plt.show()

    def plotinferredimagesfromdataset(imagedict, model, device, classes, savefilename=None, dogradcam=True,
                                      size=(10, 25),layerNo=None):

        for key, value in imagedict.items():
            fig, axes = plt.subplots(nrows=1, ncols=2, figsize=size)
            axes[0].set_title(key)
            if value.is_cuda == True:
                PlotData.sh(value.squeeze(0), axes[0])
            else:
                axes[0].imshow(value, cmap="gray", interpolation='bicubic')

            if dogradcam == True:
                gradcamimage, prediction = gradcamhelper.dogradcam(model=model, image=value, device=device,
                                                                   classes=classes,layerNo=layerNo)
                tensor = gradcamimage[0].squeeze()
                tensor = tensor.permute(1, 2, 0)
                img = tensor.cpu().numpy()
                axes[1].imshow(img, cmap="gray", interpolation='bicubic')

            if savefilename != None:
                fig.savefig("images/" + savefilename + ".png", bbox_inches='tight')



    def sh(img, ax):
        img = img / 2 + 0.5  # unnormalize
        npimg = img.cpu().numpy()
        ax.imshow(np.transpose(npimg, (1, 2, 0)))
