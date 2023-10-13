import sys, os
import torch
import argparse
import importlib
import setproctitle, socket, uuid
import datetime

from datasets import get_dataset
from models import get_model
from utils.train import train
from utils.conf import *
from utils.args import *
from utils.checkpoint import create_load_ckpt

conf_path = os.getcwd() + "."
sys.path.append(conf_path)


def parse_args():
    parser = argparse.ArgumentParser(description='Reasoning Shortcut', allow_abbrev=False)
    parser.add_argument('--model', type=str,default='cext', help='Model for inference.', choices=get_all_models())
    parser.add_argument('--load_best_args', action='store_true', help='Loads the best arguments for each method, '
                             'dataset and memory buffer.') 
    
    torch.set_num_threads(4)

    add_management_args(parser)
    args = parser.parse_known_args()[0]
    mod = importlib.import_module('models.' + args.model)

    # LOAD THE PARSER SPECIFIC OF THE MODEL, WITH ITS SPECIFICS
    get_parser = getattr(mod, 'get_parser') 
    parser = get_parser()
    parser.add_argument('--project', type=str, default="Reasoning-Shortcuts", help='wandb project')

    args = parser.parse_args() # this is the return

    # load args related to seed etc.
    set_random_seed(args.seed) if args.seed is not None else set_random_seed(42)
    
    return args

def main(args):
    
    # Add uuid, timestamp and hostname for logging
    args.conf_jobnum = str(uuid.uuid4())
    args.conf_timestamp = str(datetime.datetime.now())
    args.conf_host = socket.gethostname()
    dataset = get_dataset(args)

    # Load dataset, model, loss, and optimizer
    encoder, decoder  = dataset.get_backbone()
    n_images, c_split = dataset.get_split()
    model = get_model(args, encoder, decoder, n_images, c_split) 
    loss  = model.get_loss(args)
    model.start_optim(args)

    # SAVE A BASE MODEL OR LOAD IT, LOAD A CHECKPOINT IF PROVIDED
    model = create_load_ckpt(model, args)

    # set job name
    setproctitle.setproctitle('{}_{}_{}'.format( args.model, args.buffer_size if 'buffer_size' in args else 0, args.dataset))

    # perform posthoc evaluation/ cl training/ joint training
    print('    Chosen device:', model.device)
    if args.posthoc: pass
    else: train(model, dataset, loss, args)

    print('\n ### Closing ###')

if __name__ == '__main__':
    args = parse_args()
    
    print(args)
    
    main(args)