#!/usr/bin/env python
# coding: utf-8

# # Import Libraries

# In[1]:


import datetime

print("Model execution started at:" + datetime.datetime.today().ctime())

# In[2]:


import src.dataset.dataset as dst
import src.dataset.dataloader as dl
import src.utils.utils as utils
import src.train.train_model as train
import src.visualization.plotdata as plotdata
import src.preprocessing.preprochelper as preprochelper
from src.train.lrfinder.lrfinder import LRFinder

# get_ipython().run_line_magic('config', 'IPCompleter.greedy=True')
# get_ipython().run_line_magic('reload_ext', 'autoreload')
#
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
#       print('Select the Runtime → "Change runtime type" menu to enable a GPU accelerator, ')
#       print('and then re-execute this cell.')
#     else:
#       print(gpu_info)
#
# printgpuinfo()


# In[5]:


# def showsysteminfo():
#     from psutil import virtual_memory
#     ram_gb = virtual_memory().total / 1e9
#     ram_gb_avail = virtual_memory().available / 1e9
#     ram_gb_used = virtual_memory().active / 1e9
#     print('Your runtime has {:.1f} gigabytes of available RAM\n'.format(ram_gb))
#     print('Your runtime has {:.1f} gigabytes of free RAM\n'.format(ram_gb_avail))
#     print('Your runtime has {:.1f} gigabytes of used RAM\n'.format(ram_gb_used))
# showsysteminfo()


# In[6]:


import torch

print(torch.__version__)

# In[7]:


batch_size = 512
# compose_train, compose_test = preprochelper.PreprocHelper.getalbumentationstraintesttransforms(cifar_mean,cifar_std)
train_transforms, test_transforms = preprochelper.PreprocHelper.getpytorchtransforms((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
ds = dst.Dataset()
train_dataset = ds.gettraindataset(train_transforms)
test_dataset = ds.gettestdataset(test_transforms)

# train_dataset = ds.gettraindataset(compose_train)
# test_dataset = ds.gettestdataset(compose_test)

dataloader = dl.Cifar10Dataloader(traindataset=train_dataset, testdataset=test_dataset, batch_size=batch_size)
train_loader = dataloader.gettraindataloader()
test_loader = dataloader.gettestdataloader()

# specify the image classes
classes = ds.getclassesinCIFAR10dataset()
dataiterator = iter(train_loader)
plotdata.PlotData.showImagesfromdataset(dataiterator, classes=classes)

# In[8]:


cnn_model, device = utils.Utils.createA11CustomResnetModel()
train_model = train.TrainModel()
train_model.showmodelsummary(cnn_model)

# In[6]:


optimizer = utils.Utils.createoptimizer(cnn_model, lr=0.08, momentum=0.9, weight_decay=0, nesterov=True)
criterion = torch.nn.CrossEntropyLoss()

# In[13]:


lr_finder = LRFinder(cnn_model, optimizer, criterion, device="cuda")
lr_finder.range_test(train_loader, start_lr=0.0001, end_lr=1, num_iter=1000, step_mode="exp")
lr_finder.plot()

# In[14]:


lr_finder.reset()

# In[15]:


lr_finder.range_test(train_loader, val_loader=test_loader, start_lr=0.0001, end_lr=1, num_iter=200, step_mode="exp")

# In[16]:


lr_finder.plot(skip_end=0)

# In[17]:


lr_finder.reset()

# In[18]:


optimizer = utils.Utils.createoptimizer(cnn_model, lr=0.08, momentum=0.9, weight_decay=0, nesterov=True)
scheduler = utils.Utils.createscheduler(optimizer, mode='max', factor=0.9, patience=2,
                                        verbose=True)

# In[19]:


lr_data = []
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
epochs = 100
for epoch in range(1, epochs + 1):
    print("EPOCH:", epoch)
    train_model.train(cnn_model, device, train_loader, optimizer, 1)
    t_acc_epoch = train_model.test(model=cnn_model, device=device, test_loader=test_loader, class_correct=class_correct,
                                   class_total=class_total, epoch=epoch, lr_data=lr_data)
    scheduler.step(t_acc_epoch)
    for param_groups in optimizer.param_groups:
        print("Learning rate =", param_groups['lr'], " for epoch: ", epoch + 1)  # print LR for different epochs
        lr_data.append(param_groups['lr'])

# In[20]:


train_losses, train_acc = train_model.gettraindata()
test_losses, test_acc = train_model.gettestdata()
utils.Utils.savemodel(model=cnn_model, epoch=epochs, path="savedmodels/finalmodelwithdata.pt",
                      optimizer_state_dict=optimizer.state_dict
                      , train_losses=train_losses, train_acc=train_acc, test_acc=test_acc,
                      test_losses=test_losses, lr_data=lr_data, class_correct=class_correct, class_total=class_total)

# In[1]:


import src.utils.utils as utils
import src.preprocessing.albumentationstransforms as preprocessing

preproc = preprocessing.AlbumentaionsTransforms()
import glob
from PIL import Image
from src.utils.modelutils import *
import src.visualization.plotdata as plotdata
import src.dataset.dataset as dst
import src.dataset.dataloader as dl
import src.preprocessing.customcompose as customcompose
import src.train.train_model as train
import torch
from torch.utils.tensorboard import SummaryWriter
import torchvision

get_ipython().run_line_magic('load_ext', 'tensorboard')

# In[ ]:


print(torch.cuda.is_available())
saved_data, epoch, model_state_dict, optimizer_state_dict, train_losses, train_acc, test_losses, test_acc, test_losses, lr_data, class_correct, class_total = utils.Utils.loadmodel(
    path="savedmodels/finalmodelwithdata.pt")

# In[ ]:


model, device = utils.Utils.createmodelresnet18(model_state_dict=model_state_dict)

# In[ ]:


mean = [0.5, 0.5, 0.5]
std = [0.5, 0.5, 0.5]
preproc = preprocessing.AlbumentaionsTransforms()
train_transforms = preproc.gettraintransforms(mean, std)
test_transforms = preproc.gettesttransforms(mean, std)
compose_train = customcompose.CustomCompose(train_transforms)
compose_test = customcompose.CustomCompose(test_transforms)

ds = dst.Dataset()
train_dataset = ds.gettraindataset(compose_train)
test_dataset = ds.gettestdataset(compose_test)

batch_size = 128
dataloader = dl.Cifar10Dataloader(traindataset=train_dataset, testdataset=test_dataset, batch_size=batch_size)
test_loader = dataloader.gettestdataloader()
train_loader = dataloader.gettraindataloader()

# obtain one batch of test images
dataiterator = iter(test_loader)
# specify the image classes
classes = ds.getclassesinCIFAR10dataset()

# In[5]:


classified, misclassified = train.TrainModel.getinferredimagesfromdataset(dataiterator=dataiterator, model=model,
                                                                          classes=classes, batch_size=batch_size,
                                                                          number=25)

# In[19]:


print("Gradcam of misclassified images for Layer 34, Conv2d, Output Shape = 8")

plotdata.PlotData.plotinferredimagesfromdataset(misclassified, model, device, classes, "misclassifed"
                                                , size=(15, 20), layerNo=34)

# In[22]:


print("Gradcam of correct classified images for Layer 34, Conv2d, Output Shape = 8")

plotdata.PlotData.plotinferredimagesfromdataset(classified, model, device, classes, "correct"
                                                , size=(15, 20), layerNo=34)

# In[7]:


utils.Utils.showaccuracyacrossclasses(class_correct=class_correct, class_total=class_total)

# In[8]:


plotdata.PlotData.plottesttraingraph(train_losses=train_losses, train_acc=train_acc, test_losses=test_losses,
                                     test_acc=test_acc, lr_data=lr_data, plotonsamegraph=True, epochs=epoch,
                                     doProcessArray=False)

# In[9]:


# from src.utils.modelutils import subplot
image_paths = glob.glob('./images/testimages/*.*')
images = list(map(lambda x: Image.open(x), image_paths))
subplot(images, title='inputs', nrows=2, ncols=5)

# In[10]:


inputs = [torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)), torchvision.transforms.ToTensor(),
                                          torchvision.transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])(
    x).unsqueeze(0) for x in images]  # add 1 dim for batch
inputs = [i.to(device) for i in inputs]

# In[21]:


# specify the image classes
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']

print("Gradcam of external images for Layer 34, Conv2d, Output Shape = 8")
loc = 0
for input in inputs:
    dict = {loc: input}
    plotdata.PlotData.plotinferredimagesfromdataset(dict, model, device, classes, "external"
                                                    , size=(15, 20), layerNo=34)
    loc += 1

# In[5]:


images, labels = next(iter(train_loader))

# In[6]:


model, device = utils.Utils.createmodelresnet18(model_state_dict=model_state_dict)
images, labels = images.to(device), labels.to(device)
grid = torchvision.utils.make_grid(images)

# In[7]:


epochs = epoch

# In[8]:


writer = SummaryWriter("ReduceLR_Resnet18_albumentation_A10")
writer.add_image('images', grid, 0)
writer.add_graph(model, images)

# In[9]:


print(epochs)
for epoch in range(0, epochs):
    writer.add_scalars('Loss', {'Train': train_losses[epoch], 'Test': test_losses[epoch], }, epoch + 1)
    writer.add_scalars('Accuracy', {'Train': train_acc[epoch], 'Test': test_acc[epoch], }, epoch + 1)
    writer.add_scalar('LR', lr_data[epoch], epoch + 1)
    writer.add_histogram('Test Accuracy distribution', test_acc[epoch], epoch + 1)
    writer.add_histogram('Test Loss distribution', test_losses[epoch], epoch + 1)
    writer.add_histogram('Train Accuracy distribution', train_acc[epoch], epoch + 1)
    writer.add_histogram('Train Loss distribution', train_losses[epoch], epoch + 1)

writer.close()

# In[10]:


# tensorboard - -logdir = ReduceLR_Resnet18_albumentation_A10

# In[32]:


# torch.cuda.empty_cache()

# test_dataset = None
# train_dataset = None
# test_loader = None
# train_loader = None

# import gc
# gc.collect()


# In[ ]:
