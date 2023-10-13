import torch
import numpy as np
import itertools
import torch.nn.functional as F
from warmup_scheduler import GradualWarmupScheduler

from example.dpl_models import DPL, DPL_R
from example.xor_utils import progress_bar
from example.nesy_losses import shannon_entropy

import wandb

torch.set_printoptions(precision=3, sci_mode=False)

def train_DPL(G, Y, args):
    
    # define model
    model = DPL(3, args) 

    #define optimizer
    optim = torch.optim.Adam(model.parameters(), lr=args.lr) #lr=0.05
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=0.99)
    w_scheduler = GradualWarmupScheduler(optim, 1, 10)

    # default
    optim.zero_grad()
    optim.step()

    _loss, step, j = 1, 1, 0
    update_loss = np.zeros(10)
    update_loss[0] = 1
    while _loss > 0.015 and step < 10000:
        optim.zero_grad()
        pred, c = model(G)
        loss = F.nll_loss(pred.log(), Y)
        
        l0, l1 = loss.item(), 0
        
        if args.csup is not None:
            loss += args.gamma*F.binary_cross_entropy(torch.sigmoid(c[:args.csup]), G[:args.csup])
            l1 = loss.item() - l0
            
        if args.entropy:
            loss += args.gamma * (1- shannon_entropy(torch.sigmoid(c)))
            l2 = loss.item() - l1 - l0
            
        if args.wandb:            
            wandb.log({'y-loss':l0, 'epoch': step})
            if args.csup is not None:   wandb.log({'c-loss':l1})
            if args.entropy:            wandb.log({'h-loss':l2})
                
        # progress bar
        if step % 250 == 0:
            if j < 10: 
                w_scheduler.step()
            else:      
                scheduler.step()
            update_loss[ j % 10 ] = _loss
            j += 1
            progress_bar(step, 25000, str(step), _loss)


        # early stopping
        if np.abs(update_loss.mean() - update_loss[j % 10])/ update_loss.mean() < 0.0001:
            break

        _loss = loss.item()
        loss.backward()
        optim.step()

        scheduler.step()

        step += 1
    print('Stopped at step:', step, 'wit loss', _loss)
    return model

def train_DPL_REC(G, Y, args):
    
    # define model
    model = DPL_R(6, args) 

    #define optimizer
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=0.9)
    w_scheduler = GradualWarmupScheduler(optim, 1, 10)

    # default
    optim.zero_grad()
    optim.step()

    _loss, step, j = 1, 1, 0
    update_loss = np.zeros(10)
    update_loss[0] = 1
    while _loss > 0.01 and step < 25000:
        optim.zero_grad()
        pred, recon, c = model(G)
        loss = F.nll_loss(pred.log(), Y)
        
        l0, l1 = loss.item(), 0
        
        loss += args.gamma*F.binary_cross_entropy(recon[:,0], G[:,0])
        loss += args.gamma*F.binary_cross_entropy(recon[:,1], G[:,1])
        loss += args.gamma*F.binary_cross_entropy(recon[:,2], G[:,2])

        lrec = loss.item() - l0
        
        if args.csup is not None:
            loss += args.gamma*F.binary_cross_entropy(torch.sigmoid(c[:args.csup]), G[:args.csup])
            l1 = loss.item() - l0 - lrec
            
        if args.entropy:
            loss += args.gamma * (1- shannon_entropy(torch.sigmoid(c)))
            l2 = loss.item() - l1 - l0 - lrec
            
        if args.wandb:            
            wandb.log({'y-loss':l0, 'rec-loss':lrec, 'epoch': step})
            if args.csup is not None:   wandb.log({'c-loss':l1})
            if args.entropy:            wandb.log({'h-loss':l2})

        _loss = loss.item()
        loss.backward()
        optim.step()

        # progress bar
        if step % 250 == 0:
            if j < 10: 
                w_scheduler.step()
            else:      
                scheduler.step()
                model.update_temp()
            update_loss[ j % 10 ] = _loss
            j += 1
            progress_bar(step, 25000, str(step), _loss)

        # early stopping
        if np.abs(update_loss.mean() - update_loss[j % 10])/ update_loss.mean() < 0.0001:
            break

        step += 1

    print('Stopped at step:', step, 'wit loss', _loss)
    return model