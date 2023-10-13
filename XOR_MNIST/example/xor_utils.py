import itertools 
import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import random
from typing import Union
from datetime import datetime
from sklearn.metrics import accuracy_score, confusion_matrix
import torch.nn.functional as F
import wandb

from example.xor_metrics import extract_statistics
from example.dpl_models import DPL, DPL_R 
from example.sl_models import  SL, SL_R
from example.ltn_models import LTN, LTN_R

def set_random_seed(seed: int) -> None:
    """
    Sets the seeds at a certain value.
    :param seed: the value to be set
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def progress_bar(i: int, max_iter: int, epoch: Union[int, str],
                 loss: float) -> None:
    """
    Prints out the progress bar on the stderr file.
    :param i: the current iteration
    :param max_iter: the maximum number of iteration
    :param epoch: the epoch
    :param task_number: the task index
    :param loss: the current value of the loss function
    """
    # if not (i + 1) % 10 or (i + 1) == max_iter:
    progress = min(float((i + 1) / max_iter), 1)
    progress_bar = ('█' * int(50 * progress)) + ('┈' * (50 - int(50 * progress)))
    print('\r[ {} ] epoch {}: |{}| loss: {}'.format(
        datetime.now().strftime("%m-%d | %H:%M"),
        epoch,
        progress_bar,
        round(loss, 8)
    ), file=sys.stderr, end='', flush=True)

def show_cf(model, G, Y, args, _plot=False):

    # predict after training
    if isinstance(model, DPL):
        preds, cs = model(G)
    elif isinstance(model, DPL_R):
        preds, recon, cs = model(G)
        print('Recon Loss', F.binary_cross_entropy( recon, G).item())
        for i,j,k in itertools.product(range(2), range(2),range(2)):
            C = torch.tensor([i, j, k], dtype=torch.float, device=model.device)
            logitC = model.encoder(torch.tensor([[i, j, k]], dtype=torch.float, device=model.device))[1].detach()
            pC = logitC.sigmoid()
            
    ## SL
    elif isinstance(model, SL):
        preds, cs, pC = model(G)
    elif isinstance(model, SL_R):
        preds, recon, cs, pC = model(G)
        print('Recon Loss', F.binary_cross_entropy( recon, G).item())
        for i,j,k in itertools.product(range(2), range(2),range(2)):
            C = torch.tensor([i, j, k], dtype=torch.float, device=model.device)
            logitC = model.encoder(torch.tensor([[i, j, k]], dtype=torch.float, device=model.device))[1].detach() 
            pC = logitC.sigmoid()

    ## LTN
    elif isinstance(model, LTN):
        preds, cs, pC = model(G)
    elif isinstance(model, LTN_R):
        preds, recon, cs, pC = model(G)
        print('Recon Loss', F.binary_cross_entropy( recon, G).item())
        for i,j,k in itertools.product(range(2), range(2),range(2)):
            C = torch.tensor([i, j, k], dtype=torch.float, device=model.device)
            logitC = model.encoder(torch.tensor([[i, j, k]], dtype=torch.float, device=model.device))[1].detach() 
            pC = logitC.sigmoid()

    else: NotImplementedError()
    
    if hasattr(model, 'decoder'):
        print('decode',model.decoder( C ).detach())

    # move to numpy
    if not isinstance(model, LTN) and not isinstance(model, LTN_R):
        preds = preds.detach().argmax(dim=1).numpy()
    else: 
        preds = preds.detach().numpy()
    cs = cs.sigmoid().detach().numpy()

    G =G.detach().numpy()
    Y =Y.detach().numpy()

    yacc, cacc, stats = extract_statistics(G, Y, cs, preds)

    Gs, Ys, cpred, ypred = stats
        
    if args.wandb:
        wandb.log({'y-acc': yacc, 'c-acc':cacc} )
        wandb.log({'cf-labels': wandb.plot.confusion_matrix(
            None, Ys, ypred, class_names= [str(i) for i in range(2)] ) } )
        wandb.log({'cf-concepts': wandb.plot.confusion_matrix(
            None, Gs, cpred, class_names= [str(i) for i in range(8)])} )

    c1 = np.around(cs[:,0])
    c2 = np.around(cs[:,1])
    c3 = np.around(cs[:,2])

    print(c1,c2,c3)

    y_score =  accuracy_score(preds, Y)*100
    
    if _plot:
        print('Accuracy score in prediction:', y_score, '%')

    Gs = 4*G[:,0] + 2*G[:,1] + 1*G[:,2]
    Cs = 4*c1 + 2*c2 + 1*c3

    c_score = accuracy_score(Gs, Cs)*100
    if _plot:
        print('Accuracy score in concepts', c_score,'%')
    
    if _plot == True:
        
        fig = plt.figure(figsize=(10,3))
        CF = confusion_matrix(Gs, Cs)
        fig.add_subplot(1, 4 ,1)
        plt.imshow(CF, cmap='plasma')
        plt.xticks(range(8), ['(0,0,0)', '(0,0,1)', '(0,1,0)',
                            '(0,1,1)', '(1,0,0)', '(1,0,1)',
                            '(1,1,0)', '(1,1,1)'], rotation=90)

        plt.yticks(range(8), ['(0,0,0)', '(0,0,1)', '(0,1,0)',
                            '(0,1,1)', '(1,0,0)', '(1,0,1)',
                            '(1,1,0)', '(1,1,1)'])

        cs = [c1, c2, c3]
        for i in range(3):
            CF = confusion_matrix(G[:,i].numpy(), cs[i])

            fig.add_subplot(1, 4, 2+i)        
            plt.imshow(CF, cmap='plasma')
            plt.xticks(range(2), range(2))
            plt.yticks(range(2), range(2))

            del CF
        plt.show()

        flags = ''
        if args.disent: flags = flags+'-disent'
        if args.rec: flags = flags+'-rec'
        if args.csup is not None: flags = flags+f'-csup_{args.csup}'
        if args.entropy: flags = flags+'-entropy'
        fig.savefig(f'example/cfs/XOR-CF-{args.model}{flags}.png')

    return y_score, c_score