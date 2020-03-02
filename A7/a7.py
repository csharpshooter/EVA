import src.dataset.dataset as dst
import src.dataset.dataloader as dl
import src.preprocessing.preprocessing as preprocessing
import matplotlib.pyplot as plt
import numpy as np
import src.utils.utils as utils

# %matplotlib inline

preproc = preprocessing.Preprocessing()

train_transforms = preproc.gettraintransforms()
test_transforms = preproc.gettesttransforms()

ds = dst.Dataset()
train_dataset = ds.gettraindataset(train_transforms)
test_dataset = ds.gettestdataset(test_transforms)

dataloader = dl.Cifar10Dataloader(traindataset=train_dataset, testdataset=test_dataset)

train_loader = dataloader.gettraindataloader()
test_loader = dataloader.gettestdataloader()

# specify the image classes
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']

# obtain one batch of training images
dataiterator = iter(train_loader)
images, labels = dataiterator.next()
images = images.numpy()  # convert images to numpy for display

# plot the images in the batch, along with the corresponding labels
fig = plt.figure(figsize=(25, 4))
# display 20 images
for idx in np.arange(20):
    ax = fig.add_subplot(2, 20 / 2, idx + 1, xticks=[], yticks=[])
    utils.Utils.imshow(images[idx])
    ax.set_title(classes[labels[idx]])
