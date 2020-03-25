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

        plt.savefig("images/imagesfromdataset.png")

    def plotmisclassifiedimages(dataiterator, model, classes, batch_size, dogradcam=False, device=None):

        # plot the images in the batch, along with predicted and true labels
        fig = plt.figure(figsize=(15, 20))
        loc = 0
        while loc < 25:

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
                    ax = fig.add_subplot(5, 5, loc + 1, xticks=[], yticks=[])

                    if dogradcam != True:
                        utils.Utils.imshow(images[idx].cpu())

                        ax.set_title("Pred={} (Act={})".format(classes[preds[idx]], classes[labels[idx]])
                                     , color="red")
                    else:

                        # utils.Utils.imshow(images[idx].cpu())

                        gradcamimage, prediction = gradcamhelper.dogradcam(image=images[idx].unsqueeze(0), model=model,
                                                                           device=device,
                                                                           classes=classes)
                        # utils.Utils.imshowt(gradcamimage[0])

                        tensor = gradcamimage[0].squeeze()
                        tensor = tensor.permute(1, 2, 0)
                        # if len(tensor.shape) > 2: tensor = tensor.permute(1, 2, 0)
                        img = tensor.cpu().numpy()
                        plt.imshow(img, cmap='gray')

                        ax.set_title("Pred={} (Act={})".format(prediction["prediction"], classes[labels[idx]])
                                     , color="red")

                    loc += 1

                if loc >= 25:
                    break

            plt.savefig("images/missclassifiedimages.png")

    def plottesttraingraph(train_losses, train_acc, test_losses, test_acc, lr_data, epochs, plotonsamegraph=False,
                           doProcessArray=False):

        if doProcessArray == True:
            train_acc = utils.Utils.processarray(train_acc, epochs)
            train_losses = utils.Utils.processarray(train_losses, epochs)

        if plotonsamegraph == True:
            fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(30, 5))
            l, = axs[0].plot(train_losses, linestyle='--', label="Train Loss")
            l1, = axs[0].plot(test_losses, label="Test Loss")
            axs[0].set_title("Training and Test Loss")
            axs[0].legend(loc="best", ncol=1, handles=[l, l1])
            t, = axs[1].plot(train_acc, linestyle='--', label="Train Accuracy")
            t1, = axs[1].plot(test_acc, label="Test Accuracy")
            axs[1].set_title("Training and Test Accuracy")
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

        plt.savefig("images/traintestgraphs.png")
        plt.plot()
        plt.show()
