import os
import sys

import numpy as np
import torch
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from torch.optim.lr_scheduler import LambdaLR
from torchsummary import summary
from tqdm import tqdm

from src.train.customlossfunction.dicecoeff import dice_coeff
from src.utils import Utils


class TrainModel:

    def __init__(self):
        self.train_losses = []
        self.test_losses = []
        self.train_acc = []
        self.test_acc = []
        self.reg_loss_l1 = []
        self.factor = 0  # 0.000005
        self.loss_type = self.get_loss_function_monocular()
        self.t_acc_max = 0  # track change in validation loss
        self.optimizer = None

    def showmodelsummary(self, model, input_size=(3, 32, 32)):
        summary(model, input_size=input_size, device="cuda")

    def train(self, model, device, train_loader, optimizer, epoch):
        model.train()
        pbar = tqdm(train_loader)
        correct = 0
        processed = 0
        self.optimizer = optimizer
        for batch_idx, (data, target) in enumerate(pbar):
            # get samples
            data, target = data.to(device), target.to(device)

            # Init
            optimizer.zero_grad()
            # In PyTorch, we need to set the gradients to zero before starting to do backpropragation because PyTorch
            # accumulates the gradients on subsequent backward passes. Because of this, when you start your training
            # loop, ideally you should zero out the gradients so that you do the parameter update correctly.

            # Predict
            y_pred = model(data)

            # # Calculate L1 loss
            # l1_crit = torch.nn.L1Loss(size_average=False)
            # reg_loss = 0
            # for param in model.parameters():
            #     spare_matrix = torch.randn_like(param) * 0
            #     reg_loss += l1_crit(param, spare_matrix)
            #
            # self.reg_loss_l1.append(reg_loss)

            # Calculate loss
            loss = self.loss_type(y_pred, target)
            # loss += self.factor * reg_loss
            # self.train_losses.append(loss)

            # Backpropagation
            loss.backward()
            optimizer.step()

            # Update pbar-tqdm
            pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            processed += len(data)

            pbar.set_description(
                desc=f'Loss={loss.item()} Batch_id={batch_idx} Accuracy={100 * correct / processed:0.2f}')
        self.train_acc.append(100 * correct / processed)
        self.train_losses.append(loss)

    def test(self, model, device, test_loader, class_correct, class_total, epoch, lr_data):
        model.eval()
        test_loss = 0
        correct = 0
        t_acc = 0
        # pbar = tqdm(test_loader)
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(test_loader):
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += self.loss_type(output, target).item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct_tensor = pred.eq(target.data.view_as(pred))
                correct += pred.eq(target.view_as(pred)).sum().item()
                correct_new = np.squeeze(correct_tensor.cpu().numpy())

                # calculate test accuracy for each object class
                # for i in range(10):
                #     label = target.data[i]
                #     class_correct[label] += correct_new[i].item()
                #     class_total[label] += 1

        test_loss /= len(test_loader.dataset)
        self.test_losses.append(test_loss)

        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))

        self.test_acc.append(100. * correct / len(test_loader.dataset))
        t_acc = 100. * correct / len(test_loader.dataset)

        # save model if validation loss has decreased
        if self.t_acc_max <= t_acc:
            print('Validation accuracy increased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                self.t_acc_max,
                t_acc))
            from src.utils import Utils
            Utils.savemodel(model=model, epoch=epoch, path="savedmodels/checkpoint.pt",
                            optimizer_state_dict=self.optimizer.state_dict
                            , train_losses=self.train_losses, train_acc=self.train_acc, test_acc=self.test_acc,
                            test_losses=self.test_losses, lr_data=lr_data, class_correct=class_correct,
                            class_total=class_total)

            self.t_acc_max = t_acc

        return t_acc

    def getlossfunction(self):
        return CrossEntropyLoss()

    def get_loss_function_monocular(self):
        return BCEWithLogitsLoss()
        # return MSELoss()

    def gettraindata(self):
        return self.train_losses, self.train_acc

    def gettestdata(self):
        return self.test_losses, self.test_acc

    def getinferredimagesfromdataset(dataiterator, model, classes, batch_size, number=25):

        try:
            misclassifiedcount = 0
            classifiedcount = 0

            misclassified = {}
            classified = {}
            loop = 0

            while misclassifiedcount < number or classifiedcount < number:
                loop += 1
                # print("loop = {}".format(loop))

                img, labels = dataiterator.next()
                # images = img.numpy()

                # move model inputs to cuda
                images = img.cuda()

                # print(len(img))

                # get sample outputs
                output = model(images)
                # convert output probabilities to predicted class
                _, preds_tensor = torch.max(output, 1)
                preds = np.squeeze(preds_tensor.cpu().numpy())

                for idx in np.arange(batch_size):
                    # print("for")
                    key = "Pred={} (Act={}) ".format(classes[preds[idx]], classes[labels[idx]])

                    # print("m-" + str(misclassifiedcount))
                    # print("c-" + str(classifiedcount))
                    # print("mlen-" + str(len(misclassified)))
                    # print("clen-" + str(len(classified)))
                    # print(preds[idx])
                    # print(labels[idx].item())
                    # print(key)

                    if preds[idx] != labels[idx].item():

                        if misclassifiedcount < number:
                            key = key + str(misclassifiedcount)
                            misclassified[key] = images[idx].unsqueeze(0)
                            misclassifiedcount += 1

                    else:
                        if classifiedcount < number:
                            key = key + str(classifiedcount)
                            classified[key] = images[idx].unsqueeze(0)
                            # images[idx].cpu()
                            classifiedcount += 1

                    if misclassifiedcount >= number and classifiedcount >= number:
                        break

        except OSError as err:
            print("OS error: {0}".format(err))

        except ValueError:
            print("Could not convert data to an integer.")

        except:
            print(sys.exc_info()[0])

        return classified, misclassified

    def start_training_cyclic_lr(self, epochs, model, device, test_loader, train_loader, max_lr_epoch, weight_decay
                                 , min_lr=None,
                                 max_lr=None,
                                 cycles=1, annealing=False):
        lr_data = []
        class_correct = list(0. for i in range(10))
        class_total = list(0. for i in range(10))
        optimizer = self.get_optimizer(model=model, weight_decay=weight_decay)

        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer=optimizer, base_lr=min_lr, max_lr=max_lr,
                                                      mode='triangular2',
                                                      cycle_momentum=True, step_size_up=max_lr_epoch,
                                                      step_size_down=epochs - max_lr_epoch, )

        self.start_training(epochs, model, device, test_loader, train_loader, optimizer, scheduler, lr_data,
                            class_correct, class_total, path="savedmodels/finalmodelwithdata.pt")

        return lr_data, class_correct, class_total

    def start_training(self, epochs, model, device, test_loader, train_loader, optimizer, scheduler, lr_data,
                       class_correct, class_total, path):
        for epoch in range(0, epochs):
            print("EPOCH:", epoch)

            for param_groups in optimizer.param_groups:
                print("Learning rate =", param_groups['lr'], " for epoch: ", epoch)  # print LR for different epochs
                lr_data.append(param_groups['lr'])

            self.train(model, device, train_loader, optimizer, epoch)
            t_acc_epoch = self.test(model=model, device=device, test_loader=test_loader,
                                    class_correct=class_correct,
                                    class_total=class_total, epoch=epoch, lr_data=lr_data)
            scheduler.step()

        print('Saving final model after training cycle completion')
        self.save_model(model, epochs, optimizer.state_dict, lr_data, class_correct, class_total,
                        path=path)

        return lr_data, class_correct, class_total

    def get_optimizer(self, model, lr=1, momentum=0.9, weight_decay=0):
        optimizer = Utils.createoptimizer(model, lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=True)
        return optimizer

    def get_cyclic_scheduler(self, optimizer, epochs=25, max_lr_epoch=5, min_lr=0.01, max_lr=0.1):
        from src.train import TrainHelper
        lambda1 = TrainHelper.cyclical_lr(max_lr_epoch=max_lr_epoch, epochs=epochs, min_lr=min_lr, max_lr=max_lr)
        scheduler = LambdaLR(optimizer, lr_lambda=[lambda1])
        return scheduler

    def save_model(self, model, epochs, optimizer_state_dict, lr_data, class_correct, class_total,
                   path="savedmodels/finalmodelwithdata.pt"):
        train_losses, train_acc = self.gettraindata()
        test_losses, test_acc = self.gettestdata()
        Utils.savemodel(model=model, epoch=epochs, path=path,
                        optimizer_state_dict=optimizer_state_dict
                        , train_losses=train_losses, train_acc=train_acc, test_acc=test_acc,
                        test_losses=test_losses, lr_data=lr_data, class_correct=class_correct,
                        class_total=class_total)

    def start_training_lr_finder(self, epochs, model, device, test_loader, train_loader, lr, weight_decay, lambda_fn):
        lr_data = []
        class_correct = list(0. for i in range(10))
        class_total = list(0. for i in range(10))
        optimizer = self.get_optimizer(model=model, lr=lr, weight_decay=weight_decay)
        scheduler = Utils.create_scheduler_lambda_lr(lambda_fn, optimizer)

        return self.start_training(epochs, model, device, test_loader, train_loader, optimizer, scheduler, lr_data,
                                   class_correct, class_total, path="savedmodels/lrfinder.pt")

    def train_Monocular(self, model, device, train_loader, optimizer, epoch, loss_fn, show_output=False, infer_index=2):
        model.train()
        pbar = tqdm(train_loader)
        self.optimizer = optimizer
        iou = 0
        y_pred = None
        total_iou = 0
        train_loss = 0
        for batch_idx, (data, target) in enumerate(pbar):
            # get samples
            # data, target = data.to(device), target.to(device)

            data[0] = data[0].to(device)
            data[1] = data[1].to(device)
            data[2] = data[2].to(device)
            data[3] = data[3].to(device)

            # Init
            optimizer.zero_grad()
            # In PyTorch, we need to set the gradients to zero before starting to do backpropragation because PyTorch
            # accumulates the gradients on subsequent backward passes. Because of this, when you start your training
            # loop, ideally you should zero out the gradients so that you do the parameter update correctly.

            # Predict
            y_pred = model(data)

            # Calculate loss
            loss = loss_fn(y_pred, data[infer_index])
            iou = self.calculate_iou(data[infer_index].detach().cpu().numpy(), y_pred.detach().cpu().numpy())
            total_iou += iou
            train_loss += loss.item()
            # Backpropagation
            loss.backward()
            optimizer.step()

            # if batch_idx % 50 == 0:
            #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #         epoch, batch_idx * len(data), len(train_loader.dataset), (100. * batch_idx / len(train_loader)),
            #         loss.item()))
            #     print('IOU : {}'.format(iou))

            if batch_idx % 500 == 0:
                if show_output == True:
                    Utils.show(y_pred.detach().cpu(), nrow=4)

                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset), (100. * batch_idx / len(train_loader)),
                    loss.item()))
                print('IOU : {}'.format(iou))

        train_loss /= len(train_loader)
        total_iou /= len(total_iou)
        print('Batch IOU = {}'.format(total_iou))
        self.train_losses.append(train_loss)
        self.train_acc.append(total_iou)

        return y_pred

    def test_Monocular(self, model, device, test_loader, class_correct, class_total, epoch, lr_data, loss_fn,
                       show_output=False, infer_index=2):

        model.eval()
        test_loss = 0
        correct = 0
        pbar = tqdm(test_loader)
        output = None
        # dice_coeff_var = 0
        total_iou = 0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(pbar):
                data[0] = data[0].to(device)
                data[1] = data[1].to(device)
                data[2] = data[2].to(device)
                data[3] = data[3].to(device)
                output = model(data)

                loss = loss_fn(output, data[infer_index]).item()
                test_loss += loss
                pred = output.argmax(dim=1, keepdim=True)
                # correct += pred.eq(data[2].view_as(pred)).sum().item()

                iou = self.calculate_iou(data[infer_index].detach().cpu().numpy(), output.detach().cpu().numpy())
                total_iou += iou
                # dice_coeff_var += dice_coeff(data[1], data[infer_index]).item()

                if batch_idx % 500 == 0:
                    if show_output == True:
                        Utils.show(output.cpu(), nrow=4)

                    print('Test Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(data), len(test_loader.dataset), (100. * batch_idx / len(test_loader)),
                        loss))
                    print('IOU : {}'.format(iou))

        test_loss /= len(test_loader.dataset)
        total_iou /= len(test_loader)

        print('Batch IOU = {}'.format(total_iou))

        self.test_losses.append(test_loss)
        self.test_acc.append(total_iou)

        model_save_path = "savedmodels" + os.path.sep + "checkpoint-{}.pt".format(epoch)

        Utils.savemodel(model=model, epoch=epoch, path=model_save_path,
                        optimizer_state_dict=self.optimizer.state_dict()
                        , train_losses=self.train_losses, test_acc=self.test_acc,
                        test_losses=self.test_losses, lr_data=lr_data, class_correct=class_correct,
                        class_total=class_total)

        return output, total_iou

    def calculate_iou(self, target, prediction, thresh=0.5):
        intersection = np.logical_and(np.greater(target, thresh), np.greater(prediction, thresh))
        union = np.logical_or(np.greater(target, thresh), np.greater(prediction, thresh))
        iou_score = np.sum(intersection) / np.sum(union)
        return iou_score