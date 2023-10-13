# -*- coding: utf-8 -*-
"""
This files's functions are almost written by SENN's authors to train SENN.
We modified so as to fit the semi-supervised fashion.
"""

# standard imports
import sys
import os
import tqdm
import time
import pdb
import shutil
import torch
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import wandb

# Local imports
from SENN.utils import AverageMeter

#===============================================================================
#====================      REGULARIZER UTILITIES    ============================
#===============================================================================
"""
def compute jacobian:
Inputs: 
    x: encoder's output
    fx: prediction
    device: GPU or CPU
Return:
    J: Jacobian
NOTE: This function is not modified from original SENN
"""

#===============================================================================
#==================================   TRAINERS    ==============================
#===============================================================================
"""
def save_checkpoint: 
    save best model
Inputs: 
    state: several values in the current status
    is_best: flag whether the best model or not
    outpath: save path
Return:
    None
NOTE: This function is not modified from original SENN
"""
def save_checkpoint(state, is_best, outpath):
    if outpath == None:
        NotImplementedError('Select the output path')

    if not os.path.exists(outpath):
        os.makedirs(outpath)
    filename = os.path.join(outpath, 'checkpoint.pth.tar')
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(outpath,'model_best.pth.tar'))


"""
Class ClassificationTrainer:
    Executes train, val and test and includes many functions... 
Here is an overview of each function.
def train: 
    iterate function until last epoch, called from main function in main_cub.py, like trainer.train(...)
def train_batch: 
    only print error message
def concept_learning_loss_for_weak_supervision: 
    compute losses of known concepts and discriminator (added by Sawada)
def train_epoch: 
    train 1 epoch, called from train function
def validate: 
    validate after end of each epoch, called from train function
def test_and_save: 
    after training, this function tests by test data and save the results
def concept_error: 
    compute known concept error, called from train_epoch, validate, and test_and_save functions
def binary_accuracy: 
    compute binary accuracy of task (not use in our case, is not modified by Sawada)
def accuracy: 
    compute accyracy of task (is not modified by Sawada)
def plot_losses: 
    make figures of losses (is not modified by Sawada)
"""
class ClassificationTrainer():
    
    """
    def train: 
        iterate function until last epoch, called from main function in main_cub.py, like trainer.train(...)
    Inputs:
        train_loader: training data
        val_loader: validation data
        epochs: # of epochs for training
        save_path: save file's path
    Returns:
        None
    """    
    def train(self, train_loader, val_loader = None, epochs = 10, save_path = None):
        best_prec1 = 0

        # validate evaluation
        if val_loader is not None:
            val_prec1  = 1
            val_prec1, best_loss = self.validate(val_loader, 0)
            
        if self.args.wandb is not None:
            wandb.log({'start-lr': self.args.lr})
            
        for epoch in range(epochs):
            
            if save_path is not None:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'lr': self.args.lr,
                    'theta_reg_lambda': self.args.theta_reg_lambda,
                    'theta_reg_type': self.args.theta_reg_type,
                    'state_dict': self.model.state_dict(),
                    'best_prec1': best_prec1,
                    'optimizer' : self.optimizer.state_dict(),
                    'pretrained': [0],
                    'model': self.model
                 }, False, save_path)

            # go to train_epoch function
            self.train_epoch(epoch, train_loader, val_loader)
            
            if self.args.wandb is not None:
               wandb.log({'lr': float(self.scheduler.get_last_lr()[0])})
               
            # # validate evaluation
            if val_loader is not None:
                val_prec1  = 1
                val_prec1, last_loss = self.validate(val_loader, epoch+1)
                
            # if self.args.wandb is not None:
            #     wandb.log( {'val-loss': last_loss} )


            # remember best prec@1 and save checkpoint
            is_best = last_loss < best_loss
            best_loss = min(last_loss, best_loss)
            if save_path is not None:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'lr': self.args.lr,
                    'theta_reg_lambda': self.args.theta_reg_lambda,
                    'theta_reg_type': self.args.theta_reg_type,
                    'state_dict': self.model.state_dict(),
                    # 'best_prec1': best_prec1,
                    'min_loss': best_loss,
                    'optimizer' : self.optimizer.state_dict(),
                    'model': self.model  
                 }, is_best, save_path)

        # end message
        print('Training done')

    """
    def train_batch: 
        only print error message
    Inputs:
        None    
    Returns:
        Error Message
    """    
    def train_batch(self):
        raise NotImplemented('ClassificationTrainers must define their train_batch method!')        

    """
    def concept_learning_loss_for_weak_supervision: (added by Sawada)
        compute losses of known concepts and discriminator
    Inputs:
        inputs: output of Faster RCNN
        all_losses: loss file (saving all losses to print)
        concepts: correct concepts
        cbm: flag whether use CBM's model or not
        epoch: the number of current epoch. Current version does not use. But if you want to make process using epoch, please use it.
    Returns:
        info_loss: loss weighed adding discrminator's loss and known concept loss
        hh_labeled_list: predicted known concepts        
    """    
    def concept_learning_loss_for_weak_supervision(self, inputs, all_losses, concepts, cbm, senn, epoch):

        # compute predicted known concepts by inputs
        # real uses the discriminator's loss
        hh_labeled_list, h_x, real = self.model.conceptizer(inputs)
        concepts = concepts.to(self.device)

        
        if not senn:

            # compute losses of known concepts            
            # for i in range(21):

            #     lamb = self.c_freq[i] * (1 - self.c_freq[i])
            #     w_i  = 1 / self.c_freq[i] if concepts[0, i] == 1 else 1 / (1 - self.c_freq[i])
            #     labeled_loss = lamb * w_i * F.binary_cross_entropy(hh_labeled_list[0, i], concepts[0, i].to(self.device))
            #     for j in range(1,len(hh_labeled_list)):
            #         w_i  = 1 / self.c_freq[i] if concepts[j, i] == 1 else 1 / (1 - self.c_freq[i])
            #         labeled_loss = labeled_loss + lamb * w_i* F.binary_cross_entropy(hh_labeled_list[j,i], concepts[j, i].to(self.device))

            labeled_loss = torch.zeros([])
            if (-1) in self.args.which_c:
                labeled_loss = labeled_loss + F.binary_cross_entropy(hh_labeled_list[0], concepts[0].to(self.device))
                for j in range(1,len(hh_labeled_list)):
                    labeled_loss = labeled_loss + F.binary_cross_entropy(hh_labeled_list[j], concepts[j].to(self.device), )
                    # labeled_loss = labeled_loss + torch.nn.BCELoss() 
                    F.binary_cross_entropy(hh_labeled_list[j], concepts[j].to(self.device), )
            else:
                L = len(self.args.which_c)
                for i in range(21):
                    if i in self.args.which_c:
                        labeled_loss = F.binary_cross_entropy(hh_labeled_list[0, i], concepts[0, i].to(self.device)) / L
                        for j in range(1,len(hh_labeled_list)):
                            labeled_loss = labeled_loss + F.binary_cross_entropy(hh_labeled_list[j, i], concepts[j, i].to(self.device)) / L

            #MSE loss version for known concepts
            #labeled_loss = F.mse_loss(hh_labeled_list,concepts)
            #labeled_loss = labeled_loss*len(concepts[0])        

        if cbm: # Standard CBM does not use decoder
            info_loss = self.eta*labeled_loss
        elif senn:
            info_loss = info_loss
        else:
            info_loss += self.eta*labeled_loss
                
        if not senn:
            # save loss (only value) to the all_losses list
            all_losses['labeled_h'] = labeled_loss.data.cpu().numpy()

        # use in def train_batch (class GradPenaltyTrainer)
        return info_loss, hh_labeled_list
    
    def entropy_loss(self, pred_c, all_losses, epoch):

        # compute predicted known concepts by inputs
        # real uses the discriminator's loss
        avg_c = torch.mean(pred_c, dim=0)


        total_ent = - avg_c[0] * torch.log( avg_c[0] ) - (1-avg_c[0]) * torch.log(1-avg_c[0])
        total_ent /= np.log(2)
        for i in range(1,21):
            ent_i = - avg_c[i] * torch.log( avg_c[i] ) - (1-avg_c[i]) * torch.log(1-avg_c[i])
            ent_i /= np.log(2)

            assert ent_i <= 1 and ent_i >= 0, (ent_i, avg_c[i])
            
            total_ent += ent_i

        total_ent = total_ent / 21
        assert total_ent > 0 and total_ent < 1, total_ent 
                        
        # save loss (only value) to the all_losses list
        all_losses['entropy'] = total_ent.data.cpu().numpy()

        # use in def train_batch (class GradPenaltyTrainer)
        return (1- total_ent) * self.w_entropy

    """
    def train_epoch: 
        train 1 epoch, called from train function
    Inputs:
        epoch: the number of current epoch
        train_loader: training data
    Returns:
        None
    Outputs made in this function:
        print errors, losses of each epoch
    """        
    def train_epoch(self, epoch, train_loader, val_loader=None):

        # initialization of print's values
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        topc1 = AverageMeter()

        # switch to train mode
        self.model.train()

        end = time.time()
        
        for i, (inputs, targets, concepts) in enumerate(train_loader, 0):

            # measure data loading time
            data_time.update(time.time() - end)
            
            # get the inputs
            if self.cuda:
                inputs = inputs.cuda(self.device)
                concepts = concepts.cuda(self.device)
                targets = targets.cuda(self.device)

            # go to def train_batch (class GradPenaltyTrainer)
            outputs, loss, loss_dict, hh_labeled, pretrained_out = self.train_batch(inputs, targets, concepts, epoch)
            
            if self.args.wandb is not None:
                wandb.log(loss_dict)
                wandb.log({'step': i, 'epoch': epoch})
            
            # add to loss_history
            loss_dict['iter'] = i + (len(train_loader)*epoch)
            self.loss_history.append(loss_dict)

            # measure accuracy and record loss
            if self.nclasses > 4:
                # mainly use this line (current)
                prec1, _ = self.accuracy(outputs.data, targets.data, topk=(1, 5))
            elif self.nclasses in [3,4]:
                prec1, _ = self.accuracy(outputs.data, targets.data, topk=(1,self.nclasses))
            else:
                prec1, _ = self.binary_accuracy(outputs.data, targets.data), [100]

            # update each value of print's values
            losses.update(loss.data.cpu().numpy(), pretrained_out.size(0))
            top1.update(prec1[0], pretrained_out.size(0))
             
            if not self.args.senn:                
                # measure accuracy of concepts
                err = self.concept_error(hh_labeled.data, concepts)

                # update print's value
                topc1.update(err, pretrained_out.size(0))
                
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()


            if not self.args.senn:
                # print values of i-th iteration in the current epoch
                if i % self.print_freq == 0:
                    print('Epoch: [{0}][{1}/{2}]  '
                          'Time {batch_time.val:.2f} ({batch_time.avg:.2f})  '
                          'Loss {loss.val:.4f} ({loss.avg:.4f})  '.format(
                           epoch, i, len(train_loader), batch_time=batch_time,
                           data_time=data_time, loss=losses))
            else:
                # print values of i-th iteration in the current epoch
                if i % self.print_freq == 0:
                    print('Epoch: [{0}][{1}/{2}]  '
                          'Time {batch_time.val:.2f} ({batch_time.avg:.2f})  '
                          'Loss {loss.val:.4f} ({loss.avg:.4f})  '.format(
                           epoch, i, len(train_loader), batch_time=batch_time,
                           data_time=data_time, loss=losses))
            
            # validate evaluation
            # if val_loader is not None and ((i+1) % (len(train_loader) // cut)) == 0 :
            #     print()
            #     val_prec1 = self.validate(val_loader, (cut+1)*epoch+1+j)
            #     j += 1

            #     # remember best prec@1 and save checkpoint
            #     is_best = val_prec1 > best_prec1
            #     best_prec1 = max(val_prec1, best_prec1)
            #     if save_path is not None:
            #         save_checkpoint({
            #             'epoch': epoch + 1,
            #             'lr': self.args.lr,
            #             'theta_reg_lambda': self.args.theta_reg_lambda,
            #             'theta_reg_type': self.args.theta_reg_type,
            #             'state_dict': self.model.state_dict(),
            #             'best_prec1': best_prec1,
            #             'optimizer' : self.optimizer.state_dict(),
            #             'model': self.model  
            #          }, is_best, save_path)
                
        # optimizer's schedule update based on epoch
        self.scheduler.step(epoch) 


    """
    def validate: 
        validate after end of each epoch, called from train function
    Inputs:
        val_loader: validation data
    Returns:
        top1.avg: use whether models save or not
    Outputs made in this function:
        print errors, losses of each epoch
    NOTE: many code is the same to def train_epoch
    """        
    def validate(self, val_loader, epoch,  fold = None):

        # initialization of print's values
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        topc1 = AverageMeter()

        # switch to evaluate mode
        self.model.eval()

        end = time.time()

        loss_y, loss_c, loss_h = 0, 0, 0

        for i, (inputs, targets, concepts) in enumerate(val_loader):

            # get the inputs
            if self.cuda:
                inputs, targets, concepts = inputs.cuda(self.device), targets.cuda(self.device), concepts.cuda(self.device)

            # compute output        
            output = self.model(inputs)
            
            # prediction_criterion is defined in __init__ of "class GradPenaltyTrainer"
            pred_loss = self.prediction_criterion(output, targets)


            ################### EM ############################
        
            # save loss (only value) to the all_losses list
            loss_y += pred_loss.cpu().data.numpy() / len(val_loader)
            all_losses = {'prediction': pred_loss.cpu().data.numpy()}
                
            # compute loss of known concets and discriminator
            h_loss, hh_labeled = self.concept_learning_loss_for_weak_supervision(inputs, all_losses, concepts, self.args.cbm, self.args.senn, epoch)
            
            loss_h += self.entropy_loss(hh_labeled, all_losses, epoch).cpu().data.numpy() / len(val_loader)
            
            loss_c += h_loss.data.cpu().numpy() / len(val_loader) 

            # add to loss_history
            all_losses['iter'] = epoch
            
            #######################################################

            # measure accuracy and record loss
            if self.nclasses > 4:
                # mainly use this line (current)
                prec1, _ = self.accuracy(output.data, targets, topk=(1, 5))
            elif self.nclasses == 3:
                prec1, _ = self.accuracy(output.data, targets, topk=(1,3))
            else:
                prec1, _ = self.binary_accuracy(output.data, targets), [100]

            # update each value of print's values
            losses.update(pred_loss.data.cpu().numpy(), inputs.size(0))
            top1.update(prec1[0], inputs.size(0))

                        
            # measure accuracy of concepts
            hh_labeled, _, _ = self.model.conceptizer(inputs)
            if not self.args.senn:
                err = self.concept_error(hh_labeled.data, concepts)

                # update print's value
                topc1.update(err, inputs.size(0))
                        
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if not self.args.senn:
                # print values of i-th iteration in the current epoch
                if i % self.print_freq == 0:
                    print('Val: [{0}/{1}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                           i, len(val_loader), batch_time=batch_time, loss=losses))
            else:
                # print values of i-th iteration in the current epoch
                if i % self.print_freq == 0:
                    print('Val: [{0}/{1}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                           i, len(val_loader), batch_time=batch_time, loss=losses))
        val_loss_dict = {'iter': epoch, 'prediction': loss_y,  'labeled_h': loss_c, 'entropy': loss_h}
        
        if self.args.wandb is not None:
            wandb_dict = {}
            for key in val_loss_dict.keys():
                wandb_dict.update({ 'val-'+key: val_loss_dict[key]} )
            
            wandb.log(wandb_dict)
        
        self.val_loss_history.append(val_loss_dict)
        print()
        # top1.avg: use whether models save or not
        return top1.avg, loss_y

    
    """
    def test_and_save: 
        after training, this function tests by test data and save the results
    Inputs:
        test_loader: test data
        save_file_name: file name to save the predicted and correct concepts, predicted and correct classes...
    Returns:
        None
    Outputs made in this function:
        print errors, losses of each epoch
    NOTE: many code is the same to def train_epoch
    """        
    def test_and_save(self, test_loader, save_file_name, fold = None):

        # initialization of print's values
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        topc1 = AverageMeter()

        # switch to evaluate mode
        self.model.eval()

        end = time.time()

        # open the save file
        fp = open(save_file_name,'a')
        for i, (inputs, targets, concepts) in enumerate(test_loader):

            # get the inputs
            if self.cuda:
                inputs, targets, concepts = inputs.cuda(self.device), targets.cuda(self.device), concepts.cuda(self.device)

            # compute output
            output = self.model(inputs)
                         
            # prediction_criterion is defined in __init__ of "class GradPenaltyTrainer"
            loss = self.prediction_criterion(output, targets)

            # measure accuracy and record loss
            if self.nclasses > 4:
                # mainly use this line (current)
                prec1, _ = self.accuracy(output.data, targets, topk=(1, 5))
            elif self.nclasses == 3:
                prec1, _ = self.accuracy(output.data, targets, topk=(1,3))
            else:
                prec1, _ = self.binary_accuracy(output.data, targets), [100]

            # update each value of print's values
            losses.update(loss.data.cpu().numpy(), inputs.size(0))
            top1.update(prec1[0], inputs.size(0))

            # measure accuracy of concepts
            hh_labeled, hh, _ = self.model.conceptizer(inputs)
            if not self.args.senn:
                err = self.concept_error(hh_labeled.data, concepts)

                # update print's value
                topc1.update(err, inputs.size(0))
            
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if not self.args.senn:
                # if you use gpu, send results to cpu
                if self.cuda:
                    pred_labels = output.data.to("cpu")
                    #pred_labels = np.argmax(output.data.to("cpu"), axis=1)
                    concept_labels = hh_labeled.data.to("cpu")
                    concept_nolabels = hh.data.to("cpu")
                    attr = concepts.data.to("cpu")
                else:
                    pred_labels = output.data
                    concept_labels = hh_labeled
                    concept_nolabels = hh
                    attr = concepts

                # save to the file
                for j in range(len(targets)):
                    for k in range(len(targets[0])):
                        fp.write("%f,"%(targets[j][k]))
                        #fp.write("%d,%d,"%(targets[j][k],pred_labels[j][k]))
                    for k in range(len(pred_labels[0])):
                        fp.write("%f,"%(pred_labels[j][k]))
                    for k in range(concept_labels.shape[1]):
                        fp.write("%f,"%(concept_labels[j][k]))
                    for k in range(concept_nolabels.shape[1]):
                        fp.write("%f,"%(concept_nolabels[j][k]))
                    fp.write(",")
                    for k in range(attr.shape[1]):
                        fp.write("%f,"%(attr[j][k]))
                    fp.write("\n")

                # print values of i-th iteration
                if i % self.print_freq == 0:
                    print('Test on '+fold+': [{0}/{1}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                           i, len(test_loader), batch_time=batch_time, loss=losses))

            else:
                # if you use gpu, send results to cpu
                if self.cuda:
                    pred_labels = output.data.to("cpu")
                    #pred_labels = np.argmax(output.data.to("cpu"), axis=1)
                    concept_nolabels = hh.data.to("cpu")
                    attr = concepts.data.to("cpu")
                else:
                    pred_labels = output.data
                    concept_nolabels = hh
                    attr = concepts

                # save to the file
                for j in range(len(targets)):
                    for k in range(len(targets[0])):
                        fp.write("%f,"%(targets[j][k]))
                        #fp.write("%d,%d,"%(targets[j][k],pred_labels[j][k]))
                    for k in range(len(pred_labels[0])):
                        fp.write("%f,"%(pred_labels[j][k]))
                    for k in range(concept_nolabels.shape[1]):
                        fp.write("%f,"%(concept_nolabels[j][k]))
                    fp.write(",")
                    for k in range(attr.shape[1]):
                        fp.write("%f,"%(attr[j][k]))
                    fp.write("\n")

                
                # print values of i-th iteration
                if i % self.print_freq == 0:
                    print('Test: [{0}/{1}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                           i, len(test_loader), batch_time=batch_time, loss=losses))


        # close the save_file_name
        fp.close()

    
    
    """
    def concept_error: (added by Sawada)
        compute known concept error, called from train_epoch, validate, and test_and_save functions
    Inputs:
        output: predicted concepts
        target: correct concepts
    Returns:
        err: concept's error
    NOTE: many code is the same to def binary_accuracy
    """        
    def concept_error(self, output, target):
        err = torch.Tensor(1).fill_((output.round().eq(target)).float().mean()*100)
        err = (100.0-err.data[0])/100
        return err
    
    """
    def binary_accuracy: 
        compute binary accuracy of task (not use in our case)
    Inputs:
        output: predicted task class
        target: correct task class
    Returns:
        err: task's error
    NOTE: This function is not modified by Sawada
    """        
    def binary_accuracy(self, output, target):
        """Computes the accuracy"""
        return torch.Tensor(1).fill_((output.round().eq(target)).float().mean()*100)

    """
    def accuracy:
        compute accuracy of task
    Inputs:
        output: predicted task class
        target: correct task class
        topk: compute top1 accuracy and topk accuracy (currently k=5)
    Returns:
        res: task's error
    NOTE: This function is not modified by Sawada
    """        
    def accuracy(self, output, target, topk=(1,)):
        """Computes the precision@k for the specified values of k"""
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        #pred = pred.t()
        #correct = pred.eq(target.view(1, -1).expand_as(pred))

        correct = pred.eq(target.long())
        
        # if topk = (1,5), then, k=1 and k=5
        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

    """
    def plot_losses: 
        make figures of losses
    Inputs:
        save_path: path of save file
    Returns:
        res: None
    NOTE: This function is not modified by Sawada
    """        
    def plot_losses(self, save_path = None):
        loss_types = [k for k in self.loss_history[0].keys() if k != 'iter']
        losses = {k: [] for k in loss_types}
        iters  = []
        for e in self.loss_history:
            iters.append(e['iter'])
            for k in loss_types:
                losses[k].append(e[k])
        fig, ax = plt.subplots(1,len(loss_types), figsize = (4*len(loss_types), 5))
        if len(loss_types) == 1:
            ax = [ax] # Hacky, fix
        for i, k in enumerate(loss_types):
            ax[i].plot(iters, losses[k])
            ax[i].set_title('Loss: {}'.format(k))
            ax[i].set_xlabel('Iters')
            ax[i].set_ylabel('Loss')
        if save_path is not None:
            plt.savefig(save_path + '/training_losses.pdf', bbox_inches = 'tight', format='pdf', dpi=300)

        #### VALIDATION

        loss_types = [k for k in self.val_loss_history[0].keys() if k != 'iter']
        losses = {k: [] for k in loss_types}
        iters  = []
        for e in self.val_loss_history:
            iters.append(e['iter'])
            for k in loss_types:
                losses[k].append(e[k])
        fig, ax = plt.subplots(1,len(loss_types), figsize = (4*len(loss_types), 5))
        if len(loss_types) == 1:
            ax = [ax] # Hacky, fix
        for i, k in enumerate(loss_types):
            ax[i].plot(iters, losses[k])
            ax[i].set_title('Loss: {}'.format(k))
            ax[i].set_xlabel('Epoch')
            ax[i].set_ylabel('Loss')
        if save_path is not None:
            plt.savefig(save_path + '/validation_losses.pdf', bbox_inches = 'tight', format='pdf', dpi=300)
        #plt.show(block=False)

"""
class GradPenaltyTrainer: 
    Gradient Penalty Trainer. uses different penalty:
    || df/dx - dh/dx*theta  || (=  || dth/dx*h  || )
Here is overview of def functions
def __init__:
    initial setting
def train_batch:
    main training function of each iteration
def compute_conceptizer_jacobian:
    compute jacobian of output (not modified by Sawada)
def compute_conceptizer_jacobian_aux:
    compute jacobian of aux. output (added by Sawada)
"""        
class GradPenaltyTrainer(ClassificationTrainer):
    
    """
    def __init__:
        initial setting
        define self variable (e.g., self.gamma...)
    Inputs:
        model: (conceptizer,parametrizer,aggregator) for output of inception v.3
        model_aux: (conceptizer,parametrizer,aggregator) for aux. of inception v.3
        pretrained_model: inception v.3
        args: hyparparameters we set
        device: GPU or CPU
    Returns:
        None
    """
    def __init__(self, model, args, device):
        
        # hyparparameters used in the loss function
        self.lambd = args.theta_reg_lambda if ('theta_reg_lambda' in args) else 1e-6 # for regularization strenght
        self.eta   = args.h_labeled_param  if ('h_labeled_param' in args)  else 0.0 # for wealky supervised 
        self.gamma = args.info_hypara      if ('info_hypara' in args)      else 0.0 # for wealky supervised 
        self.w_entropy = args.w_entropy    if ('w_entropy' in args)        else 0.0

        print('self.eta:', self.eta )
        print('self.w_entropy', self.w_entropy)

        # use the gradient norm conputation
        self.norm = 2

        # set models
        self.model = model

        # weight of losses
        self.c_freq = np.load('BDD/c_freq.npy')
        self.y_freq = np.load('BDD/y_freq.npy')
        
        # others
        self.args = args
        self.cuda = args.cuda
        self.device = device

        self.nclasses = args.nclasses

        # # select prediction_criterion for task classification 
        # if args.nclasses <= 2 and args.objective == 'bce':
        #     self.prediction_criterion = F.binary_cross_entropy_with_logits
        # elif args.nclasses <= 2:# THis will be default.  and args.objective == 'bce_logits':
        #     self.prediction_criterion = F.binary_cross_entropy # NOTE: This does not do sigmoid itslef
        # elif args.objective == 'cross_entropy':
        #     self.prediction_criterion = F.cross_entropy
        # elif args.objective == 'mse':
        #     self.prediction_criterion = F.mse_loss            
        # else:
        #     #self.prediction_criterion = F.nll_loss # NOTE: To be used with output of log_softmax
        #     #BCE loss for multible labels  

        if args.model_name == 'dpl':
            print('Selected CE_for_loop')
            self.prediction_criterion = self.BCE_forloop
        elif args.model_name == 'dpl_auc':
            print('Selected CE_for_loop')
            self.prediction_criterion = self.CE_forloop
            
        self.learning_h = True

        # acumulate loss to make loss figure
        self.loss_history = []  
        self.val_loss_history = []

        # use to print error, loss
        self.print_freq = args.print_freq

        """
        select optimizer
        self.optimizer: [conceptizer, parametrizer, aggregator]
        self.aux_optimizer: [conceptizer, parametrizer, aggregator] for aux. output
        self.pre_optimizer: for pretrained model
        """
        if args.opt == 'adam':
            optim_betas = (0.9, 0.999)
            self.optimizer = optim.Adam(self.model.parameters(), lr= args.lr, betas=optim_betas)
        elif args.opt == 'rmsprop':
            self.optimizer = optim.RMSprop(self.model.parameters(), lr = args.lr)
        elif args.opt == 'sgd':
            self.optimizer = optim.SGD(self.model.parameters(), lr = args.lr, weight_decay = args.weight_decay, momentum=0.9)
            
        # set scheduler for learning rate
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.1)
        
        
    """
    def train_batch:
        training function of each batch
    Inputs:
        inputs: samples of each batch
        targets: labels of each batch
        concepts: correct concepts of each batch
        epoch: the number of current epoch. Current version does not use. 
    Returns:
        pred: task predicted results
        loss: loss for training
        all_losses: all losses for print
        hh_labeled: predicted known concepts
        pretrained_out: output of encoder (pretrained model)
    """
    def train_batch(self, inputs, targets, concepts, epoch):

        # Init
        self.optimizer.zero_grad()

        inputs, targets, concepts = Variable(inputs), Variable(targets), Variable(concepts)
            
        pred = self.model(inputs)

        # Calculate loss
        pred_loss = self.prediction_criterion(pred, targets)
        
        # save loss (only value) to the all_losses list
        all_losses = {'prediction': pred_loss.cpu().data.numpy()}
            
        # compute loss of known concets and discriminator
        h_loss, hh_labeled = self.concept_learning_loss_for_weak_supervision(inputs, all_losses, concepts, self.args.cbm, self.args.senn, epoch)

        # add entropy on concepts
        ent_loss = self.entropy_loss(hh_labeled, all_losses, epoch) 

        # total loss to train models
        loss = pred_loss + h_loss + ent_loss

        # back propagation
        loss.backward()

        # update each model
        self.optimizer.step()

        return pred, loss, all_losses, hh_labeled, inputs

    """
    def compute_conceptizer_jacobian:
        compute jacobian of output
    Inputs:
        x: output of encoder
    Returns:
        Jh: jacobian
    NOTE: This function is not modified by Sawada
    """

    def BCE_forloop(self,tar,pred):
        # for i in range(4):
        #     mask = (tar[:, i] > 1) | (tar[:, i] < 0) 
        #     assert len(tar[mask]) == 0, tar[mask] 

        #     lamb = self.y_freq[i] * (1 - self.y_freq[i])

        #     w_i  = 1 / self.y_freq[i] if pred[0, i] == 1 else 1 / (1 - self.y_freq[i]) 
        #     loss = lamb * w_i * F.binary_cross_entropy(tar[0, i], pred[0, i])

        #     for j in range(1,len(tar)):
        #         w_i  = 1 / self.y_freq[i] if pred[j, i] == 1 else 1 / (1 - self.y_freq[i])
        #         loss = loss + lamb * w_i * F.binary_cross_entropy(tar[j, i], pred[j, i])
        # return torch.zeros(())
        # # return torch.tensor(0)
        loss = F.binary_cross_entropy(tar[0, :4], pred[0, :4])
        
        for i in range(1,len(tar)):
            loss = loss + F.binary_cross_entropy(tar[i, :4], pred[i, :4])
        return loss 
        # return loss /4

    def CE_forloop(self, y_pred, y_true):
        y_trues = torch.split(y_true, 1, dim=-1)
        y_preds = torch.split(y_pred, 2, dim=-1)
    
        loss = 0
        for i in range(4):
            
            true = y_trues[i].view(-1)
            pred = y_preds[i]

            loss_i = F.nll_loss( pred.log(), true.to(torch.long) )
            loss += loss_i / 4

            assert loss_i > 0, pred.log() 
        
        return loss


        