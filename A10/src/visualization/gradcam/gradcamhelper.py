from src.utils import modelutils
from src.utils.modelutils import cifar10_postprocessing
from src.visualization.gradcam import GradCam


def dogradcam(image, model, device, classes,layerNo=None):
    # input = Compose([Resize((32, 32)), ToTensor(), cifar10_preprocessing])(image)
    # input = input.unsqueeze(0)
    model.eval()
    modules = modelutils.module2traced(model, image)

    layer = None
    if layerNo != None and layerNo < len(modules):
        layer = modules[layerNo]

    vis = GradCam(model.to(device), device, classes)
    img = vis(image, layer,
              target_class=None,
              postprocessing=cifar10_postprocessing,
              guide=False, )

    return img


# modules[34]