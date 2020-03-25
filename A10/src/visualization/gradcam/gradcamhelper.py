from torchvision.transforms import Compose, Resize, ToTensor

from src.utils.modelutils import cifar10_preprocessing, cifar10_postprocessing
from src.visualization.gradcam import GradCam


def dogradcam(image, model, device, classes):
    # input = Compose([Resize((32, 32)), ToTensor(), cifar10_preprocessing])(image)
    # input = input.unsqueeze(0)
    model.eval()
    vis = GradCam(model.to(device), device, classes)
    img = vis(image, None,
              target_class=None,
              postprocessing=cifar10_postprocessing,
              guide=False)

    return img
