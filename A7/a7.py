import src.dataset.dataset as dst
import src.dataset.dataloader as dl
import src.preprocessing.preprocessing as preprocessing
import matplotlib.pyplot as plt
import numpy as np
import src.utils.utils as utils
import src.models.train_model as train

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

cnn_model, device = utils.Utils.createmodel()
train_model = train.TrainModel()
train_model.showmodelsummary(cnn_model)
optimizer = utils.Utils.createoptimizer(cnn_model, lr=0.1, momentum=0.9, weight_decay=0)
scheduler = utils.Utils.createscheduler(optimizer, mode='max', factor=0.1, patience=5,
                                        verbose=True)

lr_data = []
epochs = 25
for epoch in range(0, epochs):
    print("EPOCH:", epoch)
    train_model.train(cnn_model, device, train_loader, optimizer, 1)
    t_acc_epoch = train_model.test(cnn_model, device, test_loader)
    scheduler.step(t_acc_epoch)
    for param_groups in optimizer.param_groups:
        print("Learning rate =", param_groups['lr'], " for epoch: ", epoch + 1)  # print LR for different epochs
        lr_data.append(param_groups['lr'])
