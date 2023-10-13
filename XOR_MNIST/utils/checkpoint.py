import torch
import os
from utils.conf import create_path 

def create_load_ckpt(model, args):
    create_path('data/runs')
    tag = 'dis' if not args.joint else 'joint'
    if args.task == 'product' and args.model in ['mnistsl', 'mnistslrec']:
        tag = tag+'-prod'
    if args.task == 'multiop':
        tag = tag+'-multiop'

    PATH = f'data/runs/{args.dataset}-{args.model}-{tag}-start.pt'
    
    if args.checkin is not None:
        model.load_state_dict(torch.load(args.checkin))
    
    elif os.path.exists(PATH):
        print('Loaded',PATH, '\n')
        model.load_state_dict(torch.load(PATH))
    else:
        print('Created',PATH, '\n')
        torch.save(model.state_dict(), PATH)

    return model        
