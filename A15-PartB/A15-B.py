from src.dataset.monocularhelper import MonocularHelper
from src.imports import *
import torch.optim.lr_scheduler

# from src.train.torchvision import collate_fn, train_one_epoch, warmup_lr_scheduler

# def run():
#     torch.multiprocessing.freeze_support()
#     print('loop')
#
# if __name__ == '__main__' or __name__ == '__mp_main__':
#     run()

helper = MonocularHelper()

# TODO
# path = helper.download_dataset(folder_path="data")

# TODO
# dict = helper.get_id_dictionary(path=path)

# TODO
# values, classes = helper.get_class_to_id_dict(id_dict=dict, path=path)

# final_output = r'/media/abhijit/DATA/Development/TSAI/EVA/MaskRCNN Dataset/OverLayedImages'
# final_output_mask = r'/media/abhijit/DATA/Development/TSAI/EVA/MaskRCNN Dataset/OverLayedMask'
# final_output_dm = r'/media/abhijit/DATA/Development/TSAI/EVA/MaskRCNN Dataset/OverLayedDepthMasks'
# bg_path = r'/media/abhijit/DATA/Development/TSAI/EVA/MaskRCNN Dataset/Background'

# final_output = r'/home/abhijit/EVARepo/MonocularDS/OverLayedImages'
# final_output_mask = r'/home/abhijit/EVARepo/MonocularDS/OverLayedMask'
# final_output_dm = r'/home/abhijit/EVARepo/MonocularDS/OverLayedDepthMasks'
# bg_path = r'/home/abhijit/EVARepo/MonocularDS/Background'

final_output = r'D:\Development\TSAI\EVA\MaskRCNN Dataset\OverLayedImages'
final_output_mask = r'D:\Development\TSAI\EVA\MaskRCNN Dataset\OverLayedMask'
final_output_dm = r'D:\Development\TSAI\EVA\MaskRCNN Dataset\OverLayedDepthMasks'
bg_path = r'/D:\Development\TSAI\EVA\MaskRCNN Dataset\Background'

train_data, train_label, test_data, test_label = helper.get_train_test_data(masks_folder=final_output_mask,
                                                                            images_folder=final_output,
                                                                            depth_masks_folder=final_output_dm,
                                                                            no_of_batches=40,
                                                                            total_images_count=400000,
                                                                            bg_folder=bg_path)

print(len(train_label))
print(len(test_label))

batch_size = 16

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


last_epoch = 1

# import os
# if os.path.exists("savedmodels/checkpoint1.pt"):
#     checkpoint, epoch, model_state_dict, optimizer_state_dict, train_losses, train_acc, test_losses, test_acc \
#         , test_losses, lr_data, class_correct, class_total = utils.Utils.loadmodel("savedmodels/checkpoint1.pt")
#     cnn_model.load_state_dict(model_state_dict)
#     last_epoch = last_epoch + checkpoint['epoch']

# torch.multiprocessing.freeze_support()
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
epochs = 20
for epoch in range(1, epochs + 1):
    print("EPOCH:", epoch)

    tr_out = train_model.train_Monocular(cnn_model, device, train_loader, optimizer, 1)
    ts_out = train_model.test_Monocular(cnn_model, device, test_loader, class_correct, class_total, epoch, lr_data)

    from src.utils.utils import Utils

    Utils.show(tr_out.detach().cpu(), nrow=4)
    Utils.show(ts_out.detach().cpu(), nrow=4)

    scheduler.step()

train_losses, train_acc = train_model.gettraindata()
test_losses, test_acc = train_model.gettestdata()
utils.Utils.savemodel(model=cnn_model, epoch=epochs, path="savedmodels/finalmodelwithdata.pt",
                      optimizer_state_dict=optimizer.state_dict
                      , train_losses=train_losses, train_acc=train_acc, test_acc=test_acc,
                      test_losses=test_losses, lr_data=lr_data, class_correct=class_correct, class_total=class_total)
