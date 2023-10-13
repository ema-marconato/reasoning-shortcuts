import torch
import numpy as np
import torch.nn.functional as F
from warmup_scheduler import GradualWarmupScheduler

from example.ltn_models import LTN, LTN_R
from example.xor_utils import progress_bar
from example.nesy_losses import sat_agg_loss, shannon_entropy

import wandb

def train_LTN(G, Y, args):
    # define model
    model = LTN(3,args) 

    #define optimizer
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=0.9)
    w_scheduler = GradualWarmupScheduler(optim, 1, 10)

    # default
    optim.zero_grad()
    optim.step()

    _loss, step, j = 1, 1, 0
    grade = 1
    update_loss = np.zeros(10)
    update_loss[0] = 1
    while _loss > 0.01 and step < 25000:
        optim.zero_grad()
        pred, logitC, pC = model(G)
        # loss = F.cross_entropy(pred, Y)
        loss = sat_agg_loss(model, 
                            pC[:,0:2], pC[:,2:4], pC[:,4:],
                            Y, grade)

        l0, l1 = loss.item(), 0
    
        if args.csup is not None:
            loss += args.gamma*F.binary_cross_entropy(torch.sigmoid(logitC[:args.csup]), G[:args.csup])
            l1 = loss.item() - l0
            
        if args.entropy:
            loss += args.gamma * (1- shannon_entropy(torch.sigmoid(logitC)))
            l2 = loss.item() - l1 - l0
            
        if args.wandb:            
            wandb.log({'y-loss':l0, 'epoch': step})
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
                if j%2 == 0: grade = (grade + 1) if grade < 8 else 8
            #     model.update_temp()
            update_loss[ j % 10 ] = _loss
            j += 1
            progress_bar(step, 25000, str(step), _loss)

        # early stopping
        if np.abs(update_loss.mean() - update_loss[j % 10])/ update_loss.mean() < 0.0001:
            break

        step += 1

    print('Stopped at step:', step, 'wit loss', _loss)
    return model

def train_LTN_R(G, Y, args):
    # define model
    model = LTN_R(3, args) 

    #define optimizer
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=0.9)
    w_scheduler = GradualWarmupScheduler(optim, 1, 10)

    # default
    optim.zero_grad()
    optim.step()

    _loss, step, j = 1, 1, 0
    grade = 1
    update_loss = np.zeros(10)
    update_loss[0] = 1
    while _loss > 0.01 and step < 5000:
        optim.zero_grad()
        pred, recon, logitC, pC = model(G)
        # loss = F.cross_entropy(pred, Y)
        loss = sat_agg_loss(model.encoder, 
                            pC[:,0:2], pC[:,2:4], pC[:,4:],
                            Y, grade)
        
        l0, l1 = loss.item(), 0
        
        loss += args.gamma*F.binary_cross_entropy(recon, G)

        lrec = loss.item() - l0
    
        if args.csup is not None:
            loss += args.gamma*F.binary_cross_entropy(torch.sigmoid(logitC[:args.csup]), G[:args.csup])
            l1 = loss.item() - l0 - lrec
            
        if args.entropy:
            loss += args.gamma * (1- shannon_entropy(torch.sigmoid(logitC)))
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
                if j%2 == 0: grade = (grade + 1) if grade < 6 else 6
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

