# #!/usr/bin/env python
# # coding: utf-8
#
# # # Import Libraries
#
# # In[1]:
#
#
# import datetime
# print("Model execution started at:" + datetime.datetime.today().ctime())
#
#
# # In[2]:
#
#
# import src
# import src.dataset.dataset as dst
# import src.dataset.dataloader as dl
# import src.preprocessing.albumentationstransforms as preprocessing
# import src.utils.utils as utils
# import src.train.train_model as train
# import src.visualization.plotdata as plotdata
# import src.preprocessing.customcompose as customcompose
# from src.train.lrfinder.lrfinder import LRFinder
#
#
# # get_ipython().run_line_magic('config', 'IPCompleter.greedy=True')
# #
# # get_ipython().run_line_magic('reload_ext', 'autoreload')
#
#
# # In[3]:
#
#
# # get_ipython().run_line_magic('autoreload', '2  # Autoreload all modules')
#
#
# # In[4]:
#
#
# def printgpuinfo():
#     gpu_info = get_ipython().getoutput('nvidia-smi')
#     gpu_info = '\n'.join(gpu_info)
#     if gpu_info.find('failed') >= 0:
#       print('Select the Runtime → "Change runtime type" menu to enable a GPU accelerator, ')
#       print('and then re-execute this cell.')
#     else:
#       print(gpu_info)
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
#       print('To enable a high-RAM runtime, select the Runtime → "Change runtime type"')
#       print('menu, and then select High-RAM in the Runtime shape dropdown. Then, ')
#       print('re-execute this cell.')
#     else:
#       print('You are using a high-RAM runtime!')
#
#
# # In[6]:
#
#
# import torch
# print(torch.__version__)
#
#
# # In[7]:
#
#
# # mean,std= utils.Utils.calculatemeanandstddeviation()
# # print("mean: " + str(mean))
# # print("std: " + str(std))
# mean=[0.5,0.5,0.5]
# std=[0.5,0.5,0.5]
# preproc = preprocessing.AlbumentaionsTransforms()
# train_transforms = preproc.gettraintransforms(mean,std)
# test_transforms = preproc.gettesttransforms(mean,std)
# compose_train = customcompose.CustomCompose(train_transforms)
# compose_test = customcompose.CustomCompose(test_transforms)
#
#
# # In[8]:
#
#
# ds = dst.Dataset()
# train_dataset = ds.gettraindataset(compose_train)
# test_dataset = ds.gettestdataset(compose_test)
#
#
# # In[9]:
#
#
# batch_size = 128
# dataloader = dl.Cifar10Dataloader(traindataset=train_dataset, testdataset=test_dataset,batch_size=batch_size)
# train_loader = dataloader.gettraindataloader()
# test_loader = dataloader.gettestdataloader()
#
#
# # In[10]:
#
#
# # specify the image classes
# classes = ds.getclassesinCIFAR10dataset()
#
# #TODO show transformed images fom dataset
# dataiterator = iter(train_loader)
# plotdata.PlotData.showImagesfromdataset(dataiterator,classes=classes)
#
#
# # In[11]:
#
#
# cnn_model, device = utils.Utils.createmodelresnet18()
# train_model = train.TrainModel()
# train_model.showmodelsummary(cnn_model)
#
#
# # In[12]:
#
#
# optimizer = utils.Utils.createoptimizer(cnn_model, lr=0.06, momentum=0.9, weight_decay=0, nesterov=True)
# criterion = torch.nn.CrossEntropyLoss()
#
#
# # In[13]:
#
#
# lr_finder = LRFinder(cnn_model, optimizer, criterion, device="cuda")
# lr_finder.range_test(train_loader, start_lr=0.0001,end_lr=1, num_iter=1000, step_mode="exp")
# lr_finder.plot()
#
#
# # In[14]:
#
#
# lr_finder.reset()
#
#
# # In[15]:
#
#
# lr_finder.range_test(train_loader, val_loader=test_loader, start_lr=0.0001,end_lr=1, num_iter=200, step_mode="exp")
#
#
# # In[16]:
#
#
# lr_finder.plot(skip_end=0)
#
#
# # In[17]:
#
#
# lr_finder.reset()
#
#
# # In[13]:
#
#
# optimizer = utils.Utils.createoptimizer(cnn_model, lr=0.08, momentum=0.9, weight_decay=0, nesterov=True)
# scheduler = utils.Utils.createscheduler(optimizer, mode='max', factor=0.9, patience=2,
#                                         verbose=True)
#
#
# # In[14]:
#
#
# lr_data = []
# class_correct = list(0. for i in range(10))
# class_total = list(0. for i in range(10))
# epochs = 100
# for epoch in range(1, epochs+1):
#     print("EPOCH:", epoch)
#     train_model.train(cnn_model, device, train_loader, optimizer, 1)
#     t_acc_epoch = train_model.test(model=cnn_model, device=device, test_loader=test_loader, class_correct=class_correct,
#                                    class_total=class_total, epoch=epoch, lr_data=lr_data)
#     scheduler.step(t_acc_epoch)
#     for param_groups in optimizer.param_groups:
#         print("Learning rate =", param_groups['lr'], " for epoch: ", epoch + 1)  # print LR for different epochs
#         lr_data.append(param_groups['lr'])
#
#
# # In[15]:
#
#
# train_losses, train_acc = train_model.gettraindata()
# test_losses, test_acc = train_model.gettestdata()
# utils.Utils.savemodel(model=cnn_model,epoch=epochs,path="savedmodels/finalmodelwithdata.pt",optimizer_state_dict=optimizer.state_dict
#                       ,train_losses=train_losses, train_acc=train_acc, test_acc=test_acc,
#                       test_losses=test_losses,lr_data=lr_data,class_correct=class_correct,class_total=class_total)


# In[7]:


import src.preprocessing.albumentationstransforms as preprocessing
import src.utils.utils as utils

preproc = preprocessing.AlbumentaionsTransforms()
import glob
from PIL import Image
from src.utils.modelutils import *
import src.visualization.plotdata as plotdata
import src.dataset.dataset as dst
import src.dataset.cifar10dataloader as dl
import src.preprocessing.customcompose as customcompose
import src.train.train_model as train

# In[2]:


print(torch.cuda.is_available())
saved_data, epoch, model_state_dict, optimizer_state_dict, train_losses, train_acc, test_losses, test_acc, test_losses, lr_data, class_correct, class_total = utils.Utils.loadmodel(
    path="savedmodels/finalmodelwithdata.pt")
print(epoch)

# In[3]:


model, device = utils.Utils.createmodelresnet18(model_state_dict=model_state_dict)

# In[4]:


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

# obtain one batch of test images
dataiterator = iter(test_loader)
# specify the image classes
classes = ds.getclassesinCIFAR10dataset()

# In[6]:


# plotdata.PlotData.plotmisclassifiedimages(dataiterator=dataiterator,model=model,classes=classes,
#                                           batch_size=batch_size,dogradcam=True,device=device)


classified, misclassified = train.TrainModel.getinferredimagesfromdataset(dataiterator=dataiterator, model=model,
                                                                          classes=classes, batch_size=batch_size,
                                                                          number=30)

print(len(classified))
print(len(misclassified))

plotdata.PlotData.plotinferredimagesfromdataset(misclassified, model, device, classes)
plotdata.PlotData.plotinferredimagesfromdataset(classified, model, device, classes)

# print(classified)
# print(misclassified)

# In[6]:


utils.Utils.showaccuracyacrossclasses(class_correct=class_correct, class_total=class_total)

# In[7]:


plotdata.PlotData.plottesttraingraph(train_losses=train_losses, train_acc=train_acc, test_losses=test_losses,
                                     test_acc=test_acc, lr_data=lr_data, plotonsamegraph=True, epochs=epoch,
                                     doProcessArray=False)

# In[8]:


# from src.utils.modelutils import subplot
image_paths = glob.glob('./images/testimages/*.*')
images = list(map(lambda x: Image.open(x), image_paths))
subplot(images, title='inputs', nrows=2, ncols=5)

# In[9]:


import torchvision

inputs = [torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)), torchvision.transforms.ToTensor(),
                                          torchvision.transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])(
    x).unsqueeze(0) for x in images]  # add 1 dim for batch
inputs = [i.to(device) for i in inputs]

# In[80]:


from src.visualization.gradcam import gradcamhelper

# specify the image classes
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']

loc = 0

for input in inputs:
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 25))
    axes[0].imshow(images[loc], cmap="gray", interpolation='bicubic')

    gradcamimage, prediction = gradcamhelper.dogradcam(model=model, image=input, device=device, classes=classes)
    tensor = gradcamimage[0].squeeze()
    tensor = tensor.permute(1, 2, 0)
    img = tensor.cpu().numpy()
    axes[1].set_title(prediction["prediction"])
    axes[1].imshow(img, cmap="gray", interpolation='bicubic')
    loc += 1

# In[81]:


torch.cuda.empty_cache()

test_dataset = None
train_dataset = None
test_loader = None
train_loader = None

import gc

gc.collect()

# In[ ]:
