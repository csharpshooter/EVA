from src.dataset.monocularhelper import MonocularHelper
from src.imports import *
import torch.optim.lr_scheduler

from src.train.torchvision import collate_fn, train_one_epoch, warmup_lr_scheduler

helper = MonocularHelper()

# TODO
# path = helper.download_dataset(folder_path="data")

# TODO
# dict = helper.get_id_dictionary(path=path)

# TODO
# values, classes = helper.get_class_to_id_dict(id_dict=dict, path=path)

final_output = r'/media/abhijit/DATA/Development/TSAI/EVA/MaskRCNN Dataset/OverLayedImages'
final_output_mask = r'/media/abhijit/DATA/Development/TSAI/EVA/MaskRCNN Dataset/OverLayedMask'
final_output_dm = r'/media/abhijit/DATA/Development/TSAI/EVA/MaskRCNN Dataset/OverLayedDepthMasks'
bg_path = r'/media/abhijit/DATA/Development/TSAI/EVA/MaskRCNN Dataset/Background'

train_data, train_label, test_data, test_label = helper.get_train_test_data(masks_folder=final_output_mask,
                                                                            images_folder=final_output,
                                                                            depth_masks_folder=final_output_dm,
                                                                            no_of_batches=40,
                                                                            total_images_count=400000,
                                                                            bg_folder=bg_path)

print(len(train_label))
print(len(test_label))

batch_size = 2

train_transforms, test_transforms = preprochelper.PreprocHelper.getpytorchtransforms(image_net_mean, image_net_std)
ds = dst.Dataset()

train_dataset = ds.get_monocular_train_dataset(train_image_data=train_data, train_image_labels=train_label,
                                               train_transforms=train_transforms)

test_dataset = ds.get_monocular_test_dataset(test_image_labels=test_label, test_image_data=test_data,
                                             test_transforms=test_transforms)

torch.manual_seed(1)

dataloader = dl.Dataloader(traindataset=train_dataset, testdataset=test_dataset, batch_size=batch_size)
train_loader = dataloader.gettraindataloader()
test_loader = dataloader.gettestdataloader()

cnn_model, device = utils.Utils.createMonocularModel()

sample = next(iter(train_loader))

imgs = sample[0][0]

# grid_tensor = torchvision.utils.make_grid(imgs, 2)
# grid_image = grid_tensor.permute(1, 2, 0)

utils.Utils.show(imgs, nrow=4)

train_model = train.TrainModel()

optimizer = utils.Utils.createoptimizer(cnn_model, lr=0.01, momentum=0.9, weight_decay=1e-5)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.01)

lr_data = []
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
epochs = 5
for epoch in range(1, epochs + 1):
    print("EPOCH:", epoch)

    if epoch == 0:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(train_loader) - 1)

        lr_scheduler = warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    train_model.train_Monocular(cnn_model, device, train_loader, optimizer, 1)
    # train_one_epoch(cnn_model, optimizer, train_loader, device, epoch, print_freq=10)
    # t_acc_epoch = train_model.test(model=cnn_model, device=device, test_loader=test_loader,
    # class_correct=class_correct, class_total=class_total, epoch=epoch, lr_data=lr_data)
    scheduler.step()
    # for param_groups in optimizer.param_groups:
    # print("Learning rate =", param_groups['lr'], " for epoch: ", epoch + 1)  # print LR for different epochs
    # lr_data.append(param_groups['lr'])

train_losses, train_acc = train_model.gettraindata()
test_losses, test_acc = train_model.gettestdata()
utils.Utils.savemodel(model=cnn_model, epoch=epochs, path="savedmodels/finalmodelwithdata.pt",
                      optimizer_state_dict=optimizer.state_dict
                      , train_losses=train_losses, train_acc=train_acc, test_acc=test_acc,
                      test_losses=test_losses, lr_data=lr_data, class_correct=class_correct, class_total=class_total)
