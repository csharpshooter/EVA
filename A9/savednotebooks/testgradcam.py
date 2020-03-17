import src.utils.utils as utils

import torch.nn as nn
import torch
import torchvision

from src.utils import ModelUtils
from src.visualization.gradcam import GradCAM

print(torch.cuda.is_available())
saved_data = utils.Utils.loadmodel(path="savedmodels/finalmodelwithdata.pt")

model, device = utils.Utils.createmodelresnet18()
model.load_state_dict(state_dict=saved_data['model_state_dict'])

# model.eval()
#
# modules = list(model.children())[:-1]
# model = nn.Sequential(*modules)
# for p in model.parameters():
#     p.requires_grad = False

imagepath = "/home/abhijit/EVARepo/EVA/A9/images/testimages/Cat-Dog.jpg"

outputdir = "/home/abhijit/EVARepo/EVA/A9/images/gradcam"

# print(model._modules.items())

import glob
from PIL import Image

image_paths = glob.glob('./images/testimages/*.*')
images = list(map(lambda x: Image.open(x), image_paths))
ModelUtils.subplot(images, title='inputs', nrows=2, ncols=5)

torchvision.transforms.Compose

inputs = [torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)), torchvision.transforms.ToTensor(),
                                          torchvision.transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])(
    x).unsqueeze(0) for x in images]  # add 1 dim for batch
inputs = [i.to(device) for i in inputs]
