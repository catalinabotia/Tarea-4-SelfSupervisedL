import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# mix precision imports
#from torch.cuda.amp import GradScaler, autocast

# custom imports
from core.misc import Metric_Logger
from core.datasets import get_supervised_dataset

class optimization():
    def __init__(self):
        self.logger = Metric_Logger()
        self.lt_logger = Metric_Logger()
        self.vt_logger = Metric_Logger()

    def train(self, model, device, optimizer, step_class,
              train_loader, use_mix_precision=False):

        self.logger.restart(len(train_loader))

        if not model.training:
            model.train()

        if use_mix_precision:
            scaler = GradScaler()

        for idx, (im1, im2) in enumerate(train_loader):

            im1 = im1.to(device)
            im2 = im2.to(device)

            if use_mix_precision:
                with autocast():
                    im1 = im1.to(device, dtype=torch.float)
                    im2 = im2.to(device, dtype=torch.float)
                    loss = step_class(phase='unsupervised',
                                      im1=im1, im2=im2, model=model)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                im1 = im1.to(device, dtype=torch.float)
                im2 = im2.to(device, dtype=torch.float)
                loss = step_class(phase='unsupervised',
                                  im1=im1, im2=im2, model=model)
                loss.backward()
                optimizer.step()
            
            optimizer.zero_grad()

            self.logger.add_metric(loss.item(), im1.size(0) * 2)
            
            self.logger.print_progress(idx)

        self.logger.print_progress(idx, True)

        return self.logger.get_mean()

    def train_linear(self, model, linear, device, optimizer, step_class,
                     train_loader, epoch=None, epochs=None):

        linear.train()

        self.lt_logger.restart(len(train_loader))

        for idx, (img, lbl) in enumerate(train_loader):

            img = img.to(device, dtype=torch.float)
            lbl = lbl.to(device, dtype=torch.long)


            loss, top1 = step_class(phase='linear',
                                    img=img, lbl=lbl,
                                    model=model, linear=linear)

            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            self.lt_logger.add_metric_and_top1(loss.item(), top1,
                                               img.size(0))
            self.lt_logger.print_linear_progress(idx, epoch, epochs)

        self.lt_logger.print_linear_progress(idx, epoch, epochs, True)

        return self.lt_logger.get_mean(), self.lt_logger.get_acc()

    def eval_linear(self, model, linear, device, step_class,
                    val_loader, epoch=None, epochs=None):

        linear.eval()

        self.vt_logger.restart(len(val_loader))

        for idx, (img, lbl) in enumerate(val_loader):

            img = img.to(device, dtype=torch.float)
            lbl = lbl.to(device, dtype=torch.long)

            loss, top1 = step_class(phase='linear test',
                                    img=img, lbl=lbl,
                                    model=model, linear=linear)

            self.vt_logger.add_metric_and_top1(loss.item(), top1,
                                               img.size(0))
            self.vt_logger.print_linear_progress(idx, epoch, epochs)

        self.vt_logger.print_linear_progress(idx, epoch, epochs, True)

        return self.vt_logger.get_mean(), self.vt_logger.get_acc()


############################
### EVALUATION FUNCTIONS ###
############################


def downstream(model, device, step_class, config, optim_utils):

    print('-' * 79)
    print('Performing linear evaluation')

    # Here you have to implement the linear training and evaluation for the downstream task
    # We give you the parameters and the output variables.

    # 1. To-do: Set model in evaluation form

    # 2. To-do: create the linear classifier model

    # 3. To-do: Load supervised dataset (get_supervised_dataset)

    # 4. To-do: Create optimizer

    # 5. To-do: Implement the downstream task:
    #       First train the linear classifier
    #       then evaluate it in the same epoch

    return top_epoch, best_train_acc, best_acc
    # Output variable description:
        # top_epoch = epoch where the best accuracy in validation was obtained
        # best_train_acc = training accuracy obtained for the epoch where the best
            # validation accuracy was obtained
        # best_acc = best validation accuracy.
