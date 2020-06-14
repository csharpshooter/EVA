# REF - https://github.com/NVIDIA/apex/blob/master/examples/imagenet/main_amp.py
# https://github.com/NVIDIA/apex/blob/02a33875970e1b555754dfc4ab85d05595d23764/tests/L1/common/main_amp.py#L81

import os
import sys

import numpy as np
import torch
from torch.autograd import Variable
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from torch.optim.lr_scheduler import LambdaLR
from torchsummary import summary
from tqdm import tqdm

from src.dataset import DataPrefetcher
from src.train.average_meter import AverageMeter
from src.utils import Utils

import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

import numpy as np

# try:
#     from apex.fp16_utils import *
#     from apex import amp, optimizers
#     from apex.multi_tensor_apply import multi_tensor_applier
# except ImportError:
#     raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")


class TrainPOC:

    def __init__(self, batch_size=16, prof=-1, local_rank=0, print_freq=2, world_size=1, use_benchmark=True,
                 is_distributed=False):
        print("")
        self.prof = prof
        self.local_rank = local_rank
        self.print_freq = print_freq
        self.world_size = world_size
        best_prec1 = 0
        opt_level = None
        torch.backends.cudnn.benchmark = use_benchmark
        self.is_distributed = is_distributed
        self.is_distributed = world_size > 1
        self.batch_size = batch_size

        if self.is_distributed:
            torch.cuda.set_device(self.local_rank)
            torch.distributed.init_process_group(backend='nccl',
                                                 init_method='env://')
            self.world_size = torch.distributed.get_world_size()

        # if channels_last:
        #     self.memory_format = torch.channels_last
        # else:
        #     self.memory_format = torch.contiguous_format

    def train(self, train_loader, model, criterion, optimizer, epoch, lr, infer_index):
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        # switch to train mode
        model.train()
        end = time.time()

        prefetcher = DataPrefetcher(train_loader)
        input, target = prefetcher.next()
        i = 0
        while input is not None:
            i += 1
            if self.prof >= 0 and i == self.prof:
                print("Profiling begun at iteration {}".format(i))
                torch.cuda.cudart().cudaProfilerStart()

            if self.prof >= 0: torch.cuda.nvtx.range_push("Body of iteration {}".format(i))

            self.adjust_learning_rate(optimizer, epoch, i, len(train_loader), lr)

            # compute output
            if self.prof >= 0: torch.cuda.nvtx.range_push("forward")
            output = model(input)
            if self.prof >= 0: torch.cuda.nvtx.range_pop()
            loss = criterion(output, input[infer_index])

            # compute gradient and do SGD step
            optimizer.zero_grad()

            if self.prof >= 0: torch.cuda.nvtx.range_push("backward")
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            if self.prof >= 0: torch.cuda.nvtx.range_pop()

            # for param in model.parameters():
            #     print(param.data.double().sum().item(), param.grad.data.double().sum().item())

            if self.prof >= 0: torch.cuda.nvtx.range_push("optimizer.step()")
            optimizer.step()
            if self.prof >= 0: torch.cuda.nvtx.range_pop()

            if i % self.print_freq == 0:
                # Every print_freq iterations, check the loss, accuracy, and speed.
                # For best performance, it doesn't make sense to print these metrics every
                # iteration, since they incur an allreduce and some host<->device syncs.

                # Measure accuracy
                prec1 = self.accuracy_iou(input[infer_index].detach().cpu().numpy(), output.detach().cpu().numpy())

                # Average loss and accuracy across processes for logging
                if self.is_distributed:
                    reduced_loss = self.reduce_tensor(loss.data)
                    prec1 = self.reduce_tensor(prec1)
                    # prec5 = self.reduce_tensor(prec5)
                else:
                    reduced_loss = loss.data

                # to_python_float incurs a host<->device sync
                losses.update(to_python_float(reduced_loss), input[infer_index].size(0))
                top1.update(to_python_float(prec1), input[infer_index].size(0))
                # top5.update(to_python_float(prec5), input.size(0))

                torch.cuda.synchronize()
                batch_time.update((time.time() - end) / self.print_freq)
                end = time.time()

                # if self.local_rank == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Speed {3:.3f} ({4:.3f})\t'
                      'Loss {loss.val:.10f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(
                    epoch, i, len(train_loader),
                    self.world_size * self.batch_size / batch_time.val,
                    self.world_size * self.batch_size / batch_time.avg,
                    batch_time=batch_time,
                    loss=losses, top1=top1))
            if self.prof >= 0: torch.cuda.nvtx.range_push("prefetcher.next()")
            input, target = prefetcher.next()
            if self.prof >= 0: torch.cuda.nvtx.range_pop()

            # Pop range "Body of iteration {}".format(i)
            if self.prof >= 0: torch.cuda.nvtx.range_pop()

            if self.prof >= 0 and i == self.prof + 10:
                print("Profiling ended at iteration {}".format(i))
                torch.cuda.cudart().cudaProfilerStop()
                quit()

        return top1

    def validate(self, val_loader, model, criterion, infer_index):
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        # switch to evaluate mode
        model.eval()

        end = time.time()

        prefetcher = DataPrefetcher(val_loader)
        input, target = prefetcher.next()
        i = 0
        while input is not None:
            i += 1

            # compute output
            with torch.no_grad():
                output = model(input)
                loss = criterion(output, input[infer_index])

            # measure accuracy and record loss
            prec1 = self.accuracy_iou(input[infer_index].detach().cpu().numpy(), output.detach().cpu().numpy())
            if self.is_distributed:
                reduced_loss = self.reduce_tensor(loss.data)
                prec1 = self.reduce_tensor(prec1)
                # prec5 = self.reduce_tensor(prec5)
            else:
                reduced_loss = loss.data

            losses.update(to_python_float(reduced_loss), input.size(0))
            top1.update(to_python_float(prec1), input.size(0))
            # top5.update(to_python_float(prec5), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # TODO:  Change timings to mirror train().
            if self.local_rank == 0 and i % self.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Speed {2:.3f} ({3:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(
                    i, len(val_loader),
                    self.world_size * self.batch_size / batch_time.val,
                    self.world_size * self.batch_size / batch_time.avg,
                    batch_time=batch_time, loss=losses,
                    top1=top1))

            input, target = prefetcher.next()

        print(' * Prec@1 {top1.avg:.3f}'
              .format(top1=top1))

        return top1.avg

    def save_checkpoint(self, state, is_best, filename='checkpoint.pth.tar'):
        torch.save(state, filename)
        if is_best:
            shutil.copyfile(filename, 'model_best.pth.tar')

    def adjust_learning_rate(self, optimizer, epoch, step, len_epoch, lr):
        """LR schedule that should yield 76% converged accuracy with batch size 256"""
        factor = epoch // 30

        if epoch >= 80:
            factor = factor + 1

        lr = lr * (0.1 ** factor)

        """Warmup"""
        if epoch < 5:
            lr = lr * float(1 + step + epoch * len_epoch) / (5. * len_epoch)

        # if(self.local_rank == 0):
        #     print("epoch = {}, step = {}, lr = {}".format(epoch, step, lr))

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def accuracy(self, output, target, topk=(1,)):
        """Computes the precision@k for the specified values of k"""
        maxk = 2
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = Variable(pred.t())
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

    def accuracy_iou(self, target, prediction, thresh=0.5):
        '''
        Calculate intersection over union value
        :param target: ground truth
        :param prediction: output predicted by model
        :param thresh: threshold
        :return: iou value
        '''
        intersection = np.logical_and(np.greater(target, thresh), np.greater(prediction, thresh))
        union = np.logical_or(np.greater(target, thresh), np.greater(prediction, thresh))
        iou_score = np.sum(intersection) / np.sum(union)
        return iou_score * 100

    def reduce_tensor(self, tensor):
        rt = tensor.clone()
        dist.all_reduce(rt, op=dist.reduce_op.SUM)
        rt /= self.world_size
        return rt

    # def main():
    #     global best_prec1, args
    #
    #     args = parse()
    #     print("opt_level = {}".format(args.opt_level))
    #     print("keep_batchnorm_fp32 = {}".format(args.keep_batchnorm_fp32), type(args.keep_batchnorm_fp32))
    #     print("loss_scale = {}".format(args.loss_scale), type(args.loss_scale))
    #
    #     print("\nCUDNN VERSION: {}\n".format(torch.backends.cudnn.version()))
    #
    #     cudnn.benchmark = True
    #     best_prec1 = 0
    #     if args.deterministic:
    #         cudnn.benchmark = False
    #         cudnn.deterministic = True
    #         torch.manual_seed(args.local_rank)
    #         torch.set_printoptions(precision=10)
    #
    #     args.distributed = False
    #     if 'WORLD_SIZE' in os.environ:
    #         args.distributed = int(os.environ['WORLD_SIZE']) > 1
    #
    #     args.gpu = 0
    #     args.world_size = 1
    #
    #     if args.distributed:
    #         args.gpu = args.local_rank
    #         torch.cuda.set_device(args.gpu)
    #         torch.distributed.init_process_group(backend='nccl',
    #                                              init_method='env://')
    #         args.world_size = torch.distributed.get_world_size()
    #
    #     assert torch.backends.cudnn.enabled, "Amp requires cudnn backend to be enabled."
    #
    #     if args.channels_last:
    #         memory_format = torch.channels_last
    #     else:
    #         memory_format = torch.contiguous_format
    #
    #     # create model
    #     if args.pretrained:
    #         print("=> using pre-trained model '{}'".format(args.arch))
    #         model = models.__dict__[args.arch](pretrained=True)
    #     else:
    #         print("=> creating model '{}'".format(args.arch))
    #         model = models.__dict__[args.arch]()
    #
    #     if args.sync_bn:
    #         import apex
    #         print("using apex synced BN")
    #         model = apex.parallel.convert_syncbn_model(model)
    #
    #     model = model.cuda().to(memory_format=memory_format)
    #
    #     # Scale learning rate based on global batch size
    #     args.lr = args.lr*float(args.batch_size*args.world_size)/256.
    #     optimizer = torch.optim.SGD(model.parameters(), args.lr,
    #                                 momentum=args.momentum,
    #                                 weight_decay=args.weight_decay)
    #
    #     # Initialize Amp.  Amp accepts either values or strings for the optional override arguments,
    #     # for convenient interoperation with argparse.
    #     model, optimizer = amp.initialize(model, optimizer,
    #                                       opt_level=args.opt_level,
    #                                       keep_batchnorm_fp32=args.keep_batchnorm_fp32,
    #                                       loss_scale=args.loss_scale
    #                                       )
    #
    #     # For distributed training, wrap the model with apex.parallel.DistributedDataParallel.
    #     # This must be done AFTER the call to amp.initialize.  If model = DDP(model) is called
    #     # before model, ... = amp.initialize(model, ...), the call to amp.initialize may alter
    #     # the types of model's parameters in a way that disrupts or destroys DDP's allreduce hooks.
    #     if args.distributed:
    #         # By default, apex.parallel.DistributedDataParallel overlaps communication with
    #         # computation in the backward pass.
    #         # model = DDP(model)
    #         # delay_allreduce delays all communication to the end of the backward pass.
    #         model = DDP(model, delay_allreduce=True)
    #
    #     # define loss function (criterion) and optimizer
    #     criterion = nn.CrossEntropyLoss().cuda()
    #
    #     # Optionally resume from a checkpoint
    #     if args.resume:
    #         # Use a local scope to avoid dangling references
    #         def resume():
    #             if os.path.isfile(args.resume):
    #                 print("=> loading checkpoint '{}'".format(args.resume))
    #                 checkpoint = torch.load(args.resume, map_location = lambda storage, loc: storage.cuda(args.gpu))
    #                 args.start_epoch = checkpoint['epoch']
    #                 best_prec1 = checkpoint['best_prec1']
    #                 model.load_state_dict(checkpoint['state_dict'])
    #                 optimizer.load_state_dict(checkpoint['optimizer'])
    #                 print("=> loaded checkpoint '{}' (epoch {})"
    #                       .format(args.resume, checkpoint['epoch']))
    #             else:
    #                 print("=> no checkpoint found at '{}'".format(args.resume))
    #         resume()
    #
    #     # Data loading code
    #     traindir = os.path.join(args.data, 'train')
    #     valdir = os.path.join(args.data, 'val')
    #
    #     if(args.arch == "inception_v3"):
    #         raise RuntimeError("Currently, inception_v3 is not supported by this example.")
    #         # crop_size = 299
    #         # val_size = 320 # I chose this value arbitrarily, we can adjust.
    #     else:
    #         crop_size = 224
    #         val_size = 256
    #
    #     train_dataset = datasets.ImageFolder(
    #         traindir,
    #         transforms.Compose([
    #             transforms.RandomResizedCrop(crop_size),
    #             transforms.RandomHorizontalFlip(),
    #             # transforms.ToTensor(), Too slow
    #             # normalize,
    #         ]))
    #     val_dataset = datasets.ImageFolder(valdir, transforms.Compose([
    #             transforms.Resize(val_size),
    #             transforms.CenterCrop(crop_size),
    #         ]))
    #
    #     train_sampler = None
    #     val_sampler = None
    #     if args.distributed:
    #         train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    #         val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    #
    #     collate_fn = lambda b: fast_collate(b, memory_format)
    #
    #     train_loader = torch.utils.data.DataLoader(
    #         train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
    #         num_workers=args.workers, pin_memory=True, sampler=train_sampler, collate_fn=collate_fn)
    #
    #     val_loader = torch.utils.data.DataLoader(
    #         val_dataset,
    #         batch_size=args.batch_size, shuffle=False,
    #         num_workers=args.workers, pin_memory=True,
    #         sampler=val_sampler,
    #         collate_fn=collate_fn)
    #
    #     if args.evaluate:
    #         validate(val_loader, model, criterion)
    #         return
    #
    #     for epoch in range(args.start_epoch, args.epochs):
    #         if args.distributed:
    #             train_sampler.set_epoch(epoch)
    #
    #         # train for one epoch
    #         train(train_loader, model, criterion, optimizer, epoch)
    #
    #         # evaluate on validation set
    #         prec1 = validate(val_loader, model, criterion)
    #
    #         # remember best prec@1 and save checkpoint
    #         if args.local_rank == 0:
    #             is_best = prec1 > best_prec1
    #             best_prec1 = max(prec1, best_prec1)
    #             save_checkpoint({
    #                 'epoch': epoch + 1,
    #                 'arch': args.arch,
    #                 'state_dict': model.state_dict(),
    #                 'best_prec1': best_prec1,
    #                 'optimizer' : optimizer.state_dict(),
    #             }, is_best)
