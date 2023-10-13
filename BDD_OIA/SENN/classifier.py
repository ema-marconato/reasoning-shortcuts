# -*- coding: utf-8 -*-
""" Code for training and evaluating Self-Explaining Neural Networks.
Copyright (C) 2018 David Alvarez-Melis <dalvmel@mit.edu>

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License,
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.
"""

from __future__ import absolute_import, division, unicode_literals

import time
import numpy as np
import copy

import torch
from torch import nn
from torch.autograd import Variable
import torch.optim as optim

from .utils import AverageMeter
from .models import SENN_FFFC


def tv_reg_loss(model):
    LAMBDA = 1e-6
    params = model.params.view(-1,10,28,28)
    reg_loss = LAMBDA * (
        torch.sum(torch.abs(params[:, :, :, :-1] - params[:, :, :, 1:])) +
        torch.sum(torch.abs(params[:, :, :-1, :] - params[:, :, 1:, :]))
    )
    return reg_loss

class PyTorchClassifier(object):
    """
    Pytorch Classifier class in the style of scikit-learn
    Classifiers include Logistic Regression and MLP

    This is ported and modified from
    https://github.com/facebookresearch/SentEval/blob/master/senteval/tools/classifier.py

    """
    def __init__(self, inputdim, nclasses, l2reg=0., batch_size=64, seed=1111,
                 cuda=False, cudaEfficient=False, nepoches=4, maxepoch=200, print_freq = 1000):
        # fix seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        print(seed)
        
        if cuda:
            torch.cuda.manual_seed(seed)
        self.cuda = cuda
        self.inputdim = inputdim
        self.nclasses = nclasses
        self.l2reg = l2reg
        self.batch_size = batch_size
        self.cudaEfficient = cudaEfficient
        self.nepoches = nepoches
        self.maxepoch = maxepoch
        self.print_freq = print_freq

    def prepare_split(self, X, y, validation_data=None, validation_split=None):
        # Preparing validation data
        assert validation_split or validation_data
        if validation_data is not None:
            trainX, trainy = X, y
            devX, devy = validation_data
        else:
            permutation = np.random.permutation(len(X))
            trainidx = permutation[int(validation_split*len(X)):]
            devidx = permutation[0:int(validation_split*len(X))]
            trainX, trainy = X[trainidx], y[trainidx]
            devX, devy = X[devidx], y[devidx]

        if self.cuda and not self.cudaEfficient:
            trainX = torch.FloatTensor(trainX).cuda()
            trainy = torch.LongTensor(trainy).cuda()
            devX = torch.FloatTensor(devX).cuda()
            devy = torch.LongTensor(devy).cuda()
        else:
            trainX = torch.FloatTensor(trainX)
            trainy = torch.LongTensor(trainy)
            devX = torch.FloatTensor(devX)
            devy = torch.LongTensor(devy)

        return trainX, trainy, devX, devy

    def fit(self, train_data, y = None, validation_data=None, validation_split=None,
            early_stop=True):
        """
            NOTE: For now I'm keeping the original fit function, which is intended
            for data given as X,y numpy objects. Maybe merge with fit_dataloader  at some point.
        """
        self.nepoch = 0
        bestaccuracy = -1
        stop_train = False
        early_stop_count = 0

        # Convert to numpy if necessary
        if isinstance(train_data, torch.utils.data.Dataset):
            X = train_data.train_data.float().numpy()
            y = train_data.train_labels.numpy()
        else:
            # Check y is provided
            X = train_data
            assert y is not None


        # Preparing validation data
        trainX, trainy, devX, devy = self.prepare_split(X, y, validation_data,
                                                        validation_split)

        # Training
        while not stop_train and self.nepoch <= self.maxepoch:
            self.trainepoch(trainX, trainy, nepoches=self.nepoches)
            accuracy = self.score(devX, devy)
            if accuracy > bestaccuracy:
                bestaccuracy = accuracy
                bestmodel = copy.deepcopy(self.model)
            elif early_stop:
                if early_stop_count >= 5:
                    stop_train = True
                early_stop_count += 1
        self.model = bestmodel
        return bestaccuracy


    def trainepoch(self, X, y, nepoches=1):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        self.model.train()
        end = time.time()
        for j in range(self.nepoch, self.nepoch + nepoches):
            permutation = np.random.permutation(len(X))
            all_costs = []
            for i in range(0, len(X), self.batch_size):
                # forward
                idx = torch.LongTensor(permutation[i:i + self.batch_size])
                if isinstance(X, torch.cuda.FloatTensor):
                    idx = idx.cuda()
                inputs = Variable(X.index_select(0, idx))
                targets = Variable(y.index_select(0, idx))
                if self.cudaEfficient:
                    inputs = inputs.cuda()
                    targets = targets.cuda()
                
                outputs = self.model(inputs.view(inputs.size(0),-1))
                # loss
                loss = self.loss_fn(outputs, targets)

                # Regularization
                if self.reg_loss is not None:
                    reg_loss = self.reg_loss(self.model)
                    loss += reg_loss


                all_costs.append(loss.data[0])
                # backward
                self.optimizer.zero_grad()
                loss.backward()
                # Update parameters
                self.optimizer.step()
                # measure accuracy and record loss
                prec1, prec5 = self.accuracy(outputs.data, targets.data, topk=(1, 5))
                losses.update(loss.data[0], inputs.size(0))
                top1.update(prec1[0], inputs.size(0))
                top5.update(prec5[0], inputs.size(0))

                 # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % self.print_freq == 0:
                    print('Epoch: [{0}][{1}/{2}]  '
                          'Time {batch_time.val:.2f} ({batch_time.avg:.2f})  '
                          #'Data {data_time.val:.2f} ({data_time.avg:.2f})  '
                          'Loss {loss.val:.4f} ({loss.avg:.4f})  '
                          'Prec@1 {top1.val:.3f} ({top1.avg:.3f})  '
                          'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                           self.nepoch + j , i, len(X), batch_time=batch_time,
                           data_time=data_time, loss=losses, top1=top1, top5=top5))

        self.nepoch += nepoches

    def score(self, devX, devy):
        self.model.eval()
        correct = 0
        if self.cuda and (not isinstance(devX, torch.cuda.FloatTensor) or self.cudaEfficient):
            devX = torch.FloatTensor(devX).cuda()
            devy = torch.LongTensor(devy).cuda()
        for i in range(0, len(devX), self.batch_size):
            Xbatch = Variable(devX[i:i + self.batch_size], volatile=True)
            ybatch = Variable(devy[i:i + self.batch_size], volatile=True)
            if self.cudaEfficient:
                Xbatch = Xbatch.cuda()
                ybatch = ybatch.cuda()
            output = self.model(Xbatch.view(Xbatch.size(0),-1))
            pred = output.data.max(1)[1]
            correct += pred.long().eq(ybatch.data.long()).sum()
        accuracy = 1.0*correct / len(devX)
        return accuracy

    def predict(self, devX):
        self.model.eval()
        if not isinstance(devX, torch.cuda.FloatTensor):
            devX = torch.FloatTensor(devX).cuda()
        yhat = np.array([])
        for i in range(0, len(devX), self.batch_size):
            Xbatch = Variable(devX[i:i + self.batch_size], volatile=True)
            output = self.model(Xbatch)
            yhat = np.append(yhat,
                             output.data.max(1)[1].cpu().numpy())
        yhat = np.vstack(yhat)
        return yhat

    def predict_proba(self, devX):
        self.model.eval()
        probas = []
        for i in range(0, len(devX), self.batch_size):
            Xbatch = Variable(devX[i:i + self.batch_size], volatile=True)
            if not probas:
                probas = self.model(Xbatch).data.cpu().numpy()
            else:
                probas = np.concatenate(probas,
                                        self.model(Xbatch).data.cpu().numpy(),
                                        axis=0)
        return probas

    def accuracy(self, output, target, topk=(1,)):
        """Computes the precision@k for the specified values of k"""
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

    def accuracy_per_class(self, model, test_loader, classes):
        """ TODO: Homogenize with accuracy style and syntax"""
        n = len(classes)
        class_correct = list(0. for i in range(n))
        class_total = list(0. for i in range(n))
        confusion_matrix = ConfusionMeter(n) #I have 2 classes here
        for data in test_loader:
            inputs, labels = data
            if self.cuda:
                    inputs, labels = inputs.cuda(), labels.cuda()
            outputs = model(Variable(inputs))
            _, predicted = torch.max(outputs.data, 1)
            c = (predicted == labels).squeeze()
            confusion_matrix.add(predicted, labels)
            for i in range(labels.size()[0]):
                label = labels[i]
                class_correct[label] += c[i]
                class_total[label] += 1

class LogReg(PyTorchClassifier):
    """ Logistic Regression with Pytorch """
    def __init__(self, inputdim, nclasses, l2reg=0., batch_size=64,
                 seed=1111, cuda = False, cudaEfficient=False):
        super(self.__class__, self).__init__(inputdim, nclasses, l2reg,
                                             batch_size, seed, cuda, cudaEfficient)
        self.cuda = cuda
        self.model = nn.Sequential(
            nn.Linear(self.inputdim, self.nclasses),
            )
        if self.cuda:
            self.model.cuda()
        self.loss_fn = nn.CrossEntropyLoss().cuda()
        self.loss_fn.size_average = False
        self.optimizer = optim.Adam(self.model.parameters(),
                                    weight_decay=self.l2reg)

class MLP(PyTorchClassifier):
    """ MLP Regression with Pytorch """
    def __init__(self, inputdim, hiddendim, nclasses, regularization = None,
                 l2reg=0., batch_size=64,
                 dropout = False, seed=1111, cuda = False, cudaEfficient=False, nepoches=1, maxepoch=10, print_freq=1000):
        super(self.__class__, self).__init__(inputdim, nclasses, l2reg,
                                             batch_size, seed, cuda, cudaEfficient,
                                             nepoches, maxepoch, print_freq)

        self.hiddendim = hiddendim
        self.drop_p    = dropout if dropout else 0

        self.model = nn.Sequential(
            nn.Linear(self.inputdim, self.hiddendim),
            nn.Dropout(p=self.drop_p),
            nn.Tanh(),
            nn.Linear(self.hiddendim, self.nclasses),
            )

        self.cuda = cuda

        if self.cuda:
            self.model.cuda()

        self.loss_fn = nn.CrossEntropyLoss().cuda()
        self.loss_fn.size_average = False

        self.optimizer = optim.Adam(self.model.parameters(),
                                    weight_decay=self.l2reg)


class SENN_MLP(PyTorchClassifier):
    def __init__(self, inputdim, hiddendim, nclasses, regularization = None,
                 l2reg=0., batch_size=64,
                 dropout = False, seed=1111, cuda = False, cudaEfficient=False, nepoches=1, maxepoch=10, print_freq=1000):
        super(self.__class__, self).__init__(inputdim, nclasses, l2reg,
                                             batch_size, seed, cuda, cudaEfficient,
                                             nepoches, maxepoch, print_freq)

        self.hiddendim = hiddendim
        self.drop_p    = dropout if dropout else 0

        self.model = nn.Sequential(
            nn.Linear(self.inputdim, self.hiddendim),
            nn.Dropout(p=self.drop_p),
            nn.Tanh(),
            nn.Linear(self.hiddendim, self.nclasses),
            )

        self.model = SENN_FFFC(self.inputdim, self.hiddendim, self.nclasses)

        self.cuda = cuda

        if self.cuda:
            self.model.cuda()

        self.loss_fn = nn.CrossEntropyLoss().cuda()
        self.loss_fn.size_average = False
        if not regularization:
            self.reg_loss = None
        elif regularization.lower() == 'tv':
            self.reg_loss = tv_reg_loss

        self.optimizer = optim.Adam(self.model.parameters(),
                                    weight_decay=self.l2reg)
