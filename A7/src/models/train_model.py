import torch
from torch.nn import CrossEntropyLoss
from torchsummary import summary
from tqdm import tqdm
import numpy as np


# import src.utils.utils as utils

class TrainModel:

    def __init__(self):
        self.train_losses = []
        self.test_losses = []
        self.train_acc = []
        self.test_acc = []
        self.reg_loss_l1 = []
        self.factor = 0
        self.loss_type = self.getlossfunction()
        self.t_acc_max = 0  # track change in validation loss

    def showmodelsummary(self, model):
        summary(model, input_size=(3, 32, 32), device="cuda")

    def train(self, model, device, train_loader, optimizer, epoch):
        model.train()
        pbar = tqdm(train_loader)
        correct = 0
        processed = 0
        for batch_idx, (data, target) in enumerate(pbar):
            # get samples
            data, target = data.to(device), target.to(device)

            # Init
            optimizer.zero_grad()
            # In PyTorch, we need to set the gradients to zero before starting to do backpropragation because PyTorch accumulates the gradients on subsequent backward passes.
            # Because of this, when you start your training loop, ideally you should zero out the gradients so that you do the parameter update correctly.

            # Predict
            y_pred = model(data)

            # Calculate L1 loss
            l1_crit = torch.nn.L1Loss(size_average=False)
            reg_loss = 0
            for param in model.parameters():
                spare_matrix = torch.randn_like(param) * 0
                reg_loss += l1_crit(param, spare_matrix)

            self.reg_loss_l1.append(reg_loss)

            # Calculate loss
            loss = self.loss_type(y_pred, target)
            loss += self.factor * reg_loss
            self.train_losses.append(loss)

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

    def test(self, model, device, test_loader, class_correct, class_total, epoch):
        model.eval()
        test_loss = 0
        correct = 0
        t_acc = 0
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
                for i in range(10):
                    label = target.data[i]
                    class_correct[label] += correct_new[i].item()
                    class_total[label] += 1

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
            Utils.savemodel(self, epoch=epoch, model=model, path="savedmodels/checkpoint.pt")

        self.t_acc_max = t_acc
        return t_acc

    def getlossfunction(self):
        return CrossEntropyLoss()

    def gettraindata(self):
        return self.train_losses, self.train_acc

    def gettestdata(self):
        return self.test_losses, self.test_acc
