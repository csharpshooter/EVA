import src.dataset.dataset as dst
import src.dataset.dataloader as dl
# import src.preprocessing.pytorchtransforms as preprocessing
import src.preprocessing.albumentationstransforms as preprocessing
import src.utils.utils as utils
import src.models.train_model as train
import numpy as np
import src.preprocessing.customcompose as customcompose

# %matplotlib inline
from src.visualization import plotdata

# def unpickle(file):
#     import pickle
#     with open(file, 'rb') as fo:
#         dict = pickle.load(fo, encoding='bytes')
#     return dict
#
# train1 = unpickle("/home/abhijit/EVARepo/EVA/A9/data/cifar-10-batches-py/data_batch_1")
# train2 = unpickle("/home/abhijit/EVARepo/EVA/A9/data/cifar-10-batches-py/data_batch_2")
# train3 = unpickle("/home/abhijit/EVARepo/EVA/A9/data/cifar-10-batches-py/data_batch_3")
# train4 = unpickle("/home/abhijit/EVARepo/EVA/A9/data/cifar-10-batches-py/data_batch_4")
# train5 = unpickle("/home/abhijit/EVARepo/EVA/A9/data/cifar-10-batches-py/data_batch_5")
# test = unpickle("/home/abhijit/EVARepo/EVA/A9/data/cifar-10-batches-py/test_batch")
# batches = unpickle("/home/abhijit/EVARepo/EVA/A9/data/cifar-10-batches-py/batches.meta")



preproc = preprocessing.AlbumentaionsTransforms()
train_transforms = preproc.gettraintransforms()
test_transforms = preproc.gettesttransforms()
compose_train = customcompose.CustomCompose(train_transforms)
compose_test = customcompose.CustomCompose(test_transforms)

ds = dst.Dataset()
train_dataset = ds.gettraindataset(compose_train)
test_dataset = ds.gettestdataset(compose_test)

batch_size = 128
dataloader = dl.Cifar10Dataloader(traindataset=train_dataset, testdataset=test_dataset, batch_size=batch_size)

train_loader = dataloader.gettraindataloader()
test_loader = dataloader.gettestdataloader()

# specify the image classes
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']

dataiterator = iter(train_loader)
plotdata.PlotData.showImagesfromdataset(dataiterator, classes=classes)

cnn_model, device = utils.Utils.createmodelresnet18()
train_model = train.TrainModel()
train_model.showmodelsummary(cnn_model)

optimizer = utils.Utils.createoptimizer(cnn_model, lr=0.01, momentum=0.9, weight_decay=0)
# scheduler = utils.Utils.createscheduler(optimizer, mode='max', factor=0.1, patience=2,
#                                         verbose=True)

lr_data = []
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
epochs = 250
for epoch in range(1, epochs):
    print("EPOCH:", epoch)
    train_model.train(cnn_model, device, train_loader, optimizer, 1)
    t_acc_epoch = train_model.test(cnn_model, device, test_loader, class_correct=class_correct,
                                   class_total=class_total, epoch=epoch)
#     scheduler.step(t_acc_epoch)
    for param_groups in optimizer.param_groups:
        print("Learning rate =", param_groups['lr'], " for epoch: ", epoch + 1)  # print LR for different epochs
        lr_data.append(param_groups['lr'])

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

print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
    100. * np.sum(class_correct) / np.sum(class_total),
    np.sum(class_correct), np.sum(class_total)))

# obtain one batch of test images
dataiterator = iter(test_loader)
plotdata.PlotData.plotmisclassifiedimages(dataiterator=dataiterator, model=cnn_model, classes=classes)

train_losses, train_acc = train_model.gettraindata()
test_losses, test_acc = train_model.gettestdata()
plotdata.PlotData.plottesttraingraph(train_losses=train_losses, train_acc=train_acc, test_losses=test_losses,
                                     test_acc=test_acc, lr_data=lr_data)

utils.Utils.savemodel(model=cnn_model, epoch=epochs, path="savedmodels/finalmodelwithdata.pt",
                      optimizer_state_dict=optimizer.state_dict
                      , train_losses=train_losses, train_acc=train_acc, test_acc=test_acc,
                      test_losses=test_losses)
