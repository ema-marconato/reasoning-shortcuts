import torch
import numpy as np
from argparse import ArgumentParser

import os, sys
import wandb

from example.dpl_train import train_DPL, train_DPL_REC 
from example.sl_train  import train_SL, train_SL_R 
from example.ltn_train import train_LTN, train_LTN_R
from example.xor_utils import show_cf, set_random_seed

def prepare_args():
    parser = ArgumentParser()
    parser.add_argument('--model', type=str, default='dpl', choices=['dpl', 'sl', 'ltn'])
    
    # mitigation strategies
    parser.add_argument('--rec', action='store_true', help='Activate RECON' )
    parser.add_argument('--csup', type=int, default=None,  help='How many concept are supervised in order.')
    parser.add_argument('--entropy', action='store_true', default=False,  help='How many concept are supervised in order.')
    parser.add_argument('--disent', action='store_true', default=False, help='Activate disentaglement')    
    parser.add_argument('--s_w', action='store_true', default=False, help='Activate shared weights')
    
    # hyperparams
    parser.add_argument('--lr', type=float, default=0.05,  help='Learning rate.')
    parser.add_argument('--gamma', type=float, default=10,  help='Weight of Mitigations.')

    # setup
    parser.add_argument('--seed', type=int, default=42,  help='Set random seed.')
    parser.add_argument('--wandb', action='store_true', default=False, help='Enable log in wandb')
    parser.add_argument('--project', type=str, default='XOR', help='Enable log in wandb')
    args = parser.parse_args()
    return args

def xor_run(args, _plot=False):
        
    # set seeds
    set_random_seed(args.seed)

    # define data
    G = np.array([[0,0,0], [0,0,1], [0,1,0],
                [0,1,1], [1,0,0], [1,0,1],
                [1,1,0],[1,1,1]])
    Y = np.array([0,1,1,0,1,0,0,1])

    G = torch.from_numpy(G).to(dtype=torch.float, device='cpu')
    Y = torch.from_numpy(Y).to(dtype=torch.long, device='cpu')
    ##
    
    if args.wandb:
        print('\n---wandb on\n')
        wandb.init(project=args.project, 
                   entity='yours', 
                   name=str(args.model),
                   config=args)

    print('---> Choice:', 'disent' if args.disent else 'joint', '+',  
         'rec' if args.rec else 'no-rec', 'and',
         'csup' if args.csup is not None else 'no-csup', 'and', 
         'entropy' if args.entropy else 'no-ent'  )

    if args.model == 'dpl':
        if not args.rec: model = train_DPL(G, Y, args)  
        else:            model = train_DPL_REC(G, Y, args) # this activates recon
    elif args.model == 'sl':
        if not args.rec: model = train_SL(G, Y, args)  
        else:            model = train_SL_R(G, Y, args)
    elif args.model == 'ltn':
        if not args.rec: model = train_LTN(G, Y, args)  
        else:            model = train_LTN_R(G, Y, args)

    show_cf(model, G, Y, args, _plot)
    
    if args.wandb:
        wandb.finish()


if __name__=='__main__':
    args = prepare_args()
    xor_run(args, True)
    
