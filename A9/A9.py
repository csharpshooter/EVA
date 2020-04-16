#!/usr/bin/env python
# coding: utf-8

# # Import Libraries

# In[1]:


import datetime

print("Model execution started at:" + datetime.datetime.today().ctime())

# In[2]:


import src.dataset.dataset as dst
import src.dataset.dataloader as dl
# import src.preprocessing.pytorchtransforms as preprocessing
import src.preprocessing.albumentationstransforms as preprocessing
import src.utils.utils as utils
import src.models.train_model as train
import src.visualization.plotdata as plotdata
import src.preprocessing.customcompose as customcompose

# get_ipython().run_line_magic('config', 'IPCompleter.greedy=True')
# get_ipython().run_line_magic('reload_ext', 'autoreload')
#
# # In[3]:
#
#
# get_ipython().run_line_magic('autoreload', '2  # Autoreload all modules')


# In[4]:


# def printgpuinfo():
#     gpu_info = get_ipython().getoutput('nvidia-smi')
#     gpu_info = '\n'.join(gpu_info)
#     if gpu_info.find('failed') >= 0:
#         print('Select the Runtime → "Change runtime type" menu to enable a GPU accelerator, ')
#         print('and then re-execute this cell.')
#     else:
#         print(gpu_info)
#
#
# # In[5]:
#
#
# def showsysteminfo():
#     from psutil import virtual_memory
#     ram_gb = virtual_memory().total / 1e9
#     print('Your runtime has {:.1f} gigabytes of available RAM\n'.format(ram_gb))
#
#     if ram_gb < 20:
#         print('To enable a high-RAM runtime, select the Runtime → "Change runtime type"')
#         print('menu, and then select High-RAM in the Runtime shape dropdown. Then, ')
#         print('re-execute this cell.')
#     else:
#         print('You are using a high-RAM runtime!')


# In[6]:


import torch

print(torch.__version__)

# In[7]:


# mean,std= utils.Utils.calculatemeanandstddeviation()
# print("mean: " + str(mean))
# print("std: " + str(std))
mean = [0.5, 0.5, 0.5]
std = [0.5, 0.5, 0.5]
preproc = preprocessing.AlbumentaionsTransforms()
train_transforms = preproc.gettraintransforms(mean, std)
test_transforms = preproc.gettesttransforms(mean, std)
compose_train = customcompose.CustomCompose(train_transforms)
compose_test = customcompose.CustomCompose(test_transforms)

# In[8]:


ds = dst.Dataset()
train_dataset = ds.gettraindataset(compose_train)
test_dataset = ds.gettestdataset(compose_test)

# In[9]:


batch_size = 64
dataloader = dl.Cifar10Dataloader(traindataset=train_dataset, testdataset=test_dataset, batch_size=batch_size)
train_loader = dataloader.gettraindataloader()
test_loader = dataloader.gettestdataloader()

# In[10]:


# specify the image classes
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']

# TODO show transformed images fom dataset
dataiterator = iter(train_loader)
plotdata.PlotData.showImagesfromdataset(dataiterator, classes=classes)

# In[11]:


cnn_model, device = utils.Utils.createmodelresnet18()
train_model = train.TrainModel()
train_model.showmodelsummary(cnn_model)

# In[12]:


optimizer = utils.Utils.createoptimizer(cnn_model, lr=0.1, momentum=0.9, weight_decay=0.0001)
# scheduler = utils.Utils.createscheduler(optimizer, mode='max', factor=0.1, patience=4,
#                                         verbose=True)
scheduler = utils.Utils.createschedulersteplr(optimizer, step_size=25, gamma=0.5)

# In[13]:


lr_data = []
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
epochs = 75
for epoch in range(1, epochs + 1):
    print("EPOCH:", epoch)
    train_model.train(cnn_model, device, train_loader, optimizer, 1)
    t_acc_epoch = train_model.test(cnn_model, device, test_loader, class_correct=class_correct,
                                   class_total=class_total, epoch=epoch)
    scheduler.step(epoch)
    for param_groups in optimizer.param_groups:
        print("Learning rate =", param_groups['lr'], " for epoch: ", epoch + 1)  # print LR for different epochs
        lr_data.append(param_groups['lr'])

# In[14]:


import numpy as np

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

# In[15]:


# obtain one batch of test images
dataiterator = iter(test_loader)
plotdata.PlotData.plotmisclassifiedimages(dataiterator=dataiterator, model=cnn_model, classes=classes)

# In[16]:


train_losses, train_acc = train_model.gettraindata()
test_losses, test_acc = train_model.gettestdata()
plotdata.PlotData.plottesttraingraph(train_losses=train_losses, train_acc=train_acc, test_losses=test_losses,
                                     test_acc=test_acc, lr_data=lr_data)

# In[17]:


utils.Utils.savemodel(model=cnn_model, epoch=epochs, path="savedmodels/finalmodelwithdata.pt",
                      optimizer_state_dict=optimizer.state_dict
                      , train_losses=train_losses, train_acc=train_acc, test_acc=test_acc,
                      test_losses=test_losses)

# In[18]:


import torch
import src.utils.utils as utils

print(torch.cuda.is_available())
saved_data = utils.Utils.loadmodel(path="savedmodels/finalmodelwithdata.pt")

# In[19]:


model, device = utils.Utils.createmodelresnet18()
model.load_state_dict(state_dict=saved_data['model_state_dict'])

# In[20]:


import glob
from PIL import Image
from src.utils.modelutils import ModelUtils

image_paths = glob.glob('./images/testimages/*.*')
images = list(map(lambda x: Image.open(x), image_paths))
ModelUtils.subplot(images, title='inputs', nrows=2, ncols=5)

# In[21]:


import torchvision

inputs = [torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)), torchvision.transforms.ToTensor(),
                                          torchvision.transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])(
    x).unsqueeze(0) for x in images]  # add 1 dim for batch
inputs = [i.to(device) for i in inputs]

# In[22]:


from src.visualization.gradcam.gradcam import gradcamof

# specify the image classes
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']

gradcamof(model, inputs, classes)

# In[ ]:
