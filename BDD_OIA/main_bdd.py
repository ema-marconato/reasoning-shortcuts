# -*- coding: utf-8 -*-
# Standard Imports
import sys, os
import numpy as np
import pdb
import pickle
import argparse
import operator
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Torch-related
import torch
from torch.utils.data import TensorDataset
from torch.autograd import Variable
import torchvision
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
import torch.utils.data.dataloader as dataloader


# Local imports
from SENN.utils import plot_theta_stability, generate_dir_names, noise_stability_plots, concept_grid
from SENN.eval_utils import estimate_dataset_lipschitz
from SENN.arglist import get_senn_parser


from BDD.dataset import load_data, find_class_imbalance
from BDD.config import BASE_DIR, N_CLASSES, N_ATTRIBUTES, UPWEIGHT_RATIO, MIN_LR, LR_DECAY_SIZE

from models import GSENN
from DPL.dpl import DPL
from DPL.dpl_auc import DPL_AUC
from conceptizers_BDD import image_fcc_conceptizer, image_cnn_conceptizer
from parametrizers import image_parametrizer, dfc_parametrizer
from aggregators_BDD import additive_scalar_aggregator, CBM_aggregator
from trainers_BDD import GradPenaltyTrainer
import wandb


# This function does not modification
def parse_args():
    senn_parser = get_senn_parser()

    ### Local ones
    parser = argparse.ArgumentParser(parents =[senn_parser],add_help=False,
        description='Interpteratbility robustness evaluation')

    # #setup
    parser.add_argument('-d','--datasets', nargs='+',
                        default = ['heart', 'ionosphere', 'breast-cancer','wine','heart',
                        'glass','diabetes','yeast','leukemia','abalone'], help='<Required> Set flag')
    parser.add_argument('--lip_calls', type=int, default=10,
                        help='ncalls for bayes opt gp method in Lipschitz estimation')
    parser.add_argument('--lip_eps', type=float, default=0.01,
                        help='eps for Lipschitz estimation')
    parser.add_argument('--lip_points', type=int, default=100,
                        help='sample size for dataset Lipschitz estimation')
    parser.add_argument('--optim', type=str, default='gp',
                        help='black-box optimization method')
    
    parser.add_argument('--model_name', type=str, default='dpl', 
                        help='Choose model to fit')
    
    parser.add_argument('--which_c',type=int, nargs='+', default=[-1], help='Which concepts explicitly supervise (-1 means all)')
    parser.add_argument('--wandb', type=str, default=None, help='Activate wandb')
    parser.add_argument('--project', type=str, default='BDD-OIA', help='Select wandb project')
    

    #####

    args = parser.parse_args()

    print("\nParameters:")
    for attr, value in sorted(args.__dict__.items()):
        print("\t{}={}".format(attr.upper(), value))

    return args


"""
Main function:
load data, set models, train and test, and save results.
After ending this function, you can see <./out/bdd/*> directory to check outputs.

Inputs:
    None
Returns:
    None

Inputs loaded in this function:
    ./data/BDD: images of CUB_200_2011
    ./data/BDD/train_BDD_OIA.pkl, val_BDD_OIA.pkl, test_BDD_OIA.pkl: train, val, test samples
    ./models/bdd100k_24.pth: Faster RCNN pretrained by BDD100K (RCNN_global())

Outputs made in this function (same as CUB):
    *.pkl: model
    grad*/training_losses.pdf: loss figure
    grad*/concept_grid.pdf: images which maximize and minimize each unit in the concept layer
    grad*/test_results_of_BDD.csv: predicted and correct labels, prSedicted and correct concepts, coefficient of each concept
"""
def main(args):
    
    # get hyperparameters
    if args.wandb is not None:
        print('\n---wandb on\n')
        wandb.init(project=args.project, entity=args.wandb, 
                   name=str(args.model_name),
                   config=args)
        
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    
    # set which GPU uses
    if args.cuda:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   
    else:
        device = torch.device("cpu")  

    # load dataset
    train_data_path = "data/bdd2048/train_BDD_OIA.pkl"
    val_data_path   = "data/bdd2048/val_BDD_OIA.pkl"
    test_data_path  = "data/bdd2048/test_BDD_OIA.pkl"
        
    
    # load_data. Detail is BDD/dataset.py, lines 149-. This function is made by CBM's authors

    image_dir = 'data/bdd2048/'
    train_loader = load_data([train_data_path], True, False, args.batch_size, uncertain_label=False, n_class_attr=2, 
                             image_dir=image_dir+"train", 
                             resampling=False)
    valid_loader = load_data([val_data_path], True, False, args.batch_size, uncertain_label=False, n_class_attr=2,   
                             image_dir=image_dir+"val", 
                             resampling=False)
    test_loader  = load_data([test_data_path], True, False, args.batch_size, uncertain_label=False, n_class_attr=2,  
                             image_dir=image_dir+"test",  
                             resampling=False)
    
    # get paths (see SENN/utils.py, lines 34-). This function is made by SENN's authors
    model_path, log_path, results_path = generate_dir_names('bdd', args)
    
    # initialize the csv file (cleaning before training)
    save_file_name = "%s/test_results_of_BDD.csv"%(results_path)
    fp = open(save_file_name,'w')
    fp.close()

    #### EM
    save_file_name_train = "%s/train_results_of_BDD.csv"%(results_path)
    fp_train = open(save_file_name_train,'w')
    fp_train.close()

    # Convert the arguments to a string representation
    arg_string = '\n'.join([f'{arg}={getattr(args, arg)}' for arg in vars(args)])
    file_path = "%s/args.txt"%(results_path)
    with open(file_path, 'w') as f:
        f.write(arg_string)

    """
    Next, we set four networks, conceptizer, parametrizer, aggregator, and pretrained_model
    Pretrained_model (h(x)): encoder (h) Faster RCNN (see ./BDD/template_model.py)
    Conceptizer (e1(h(x))): concepts layer (see conceptizer.py)
    Parametrizer (e2(h(x))): network to compute parameters to get concepts (see parametrizer.py)
    Aggregator (f(e1(h(x)),e2(h(x)))): output layer (see aggregators.py)
    """    
    
    # only "fcc" conceptizer use, otherwise cannot use (not modifile so as to fit this task...)
    if args.h_type == "fcc":
        conceptizer1  = image_fcc_conceptizer(2048, args.nconcepts, args.nconcepts_labeled, args.concept_dim, args.h_sparsity, args.senn)
    elif args.h_type == 'cnn':
        print("[ERROR] please use fcc network")
        sys.exit(1)
    else:
        print("[ERROR] please use fcc network")
        sys.exit(1)
        

    parametrizer1 = dfc_parametrizer(2048,1024,512,256,128,args.nconcepts, args.theta_dim, layers=4)
    buf = 1

    """
    If you train CBM model, set cbm, <python main_cub.py --cbm>.
    In this case, our model does not use unknown concepts even if you set the number of unknown concepts.
    NOTE: # of unknown concepts = args.nconcepts - args.nconcepts_labeled
    """    
    if args.cbm == True:
        aggregator = CBM_aggregator(args.concept_dim, args.nclasses, args.nconcepts_labeled)
    else:
        aggregator = additive_scalar_aggregator(args.concept_dim, args.nclasses)

    # you should set load_model as True. If you set, you can use inception v.3 as the encoder, otherwise end.

    """
    Function GSENN is in models.py
    model: model using outputs of inception v.3
    model_aux: mdoel using auxiliary output of inception v.3
    """
    # model = GSENN(conceptizer1, parametrizer1, aggregator, args.cbm, args.senn) 
    if args.model_name == 'dpl':
        model = DPL(conceptizer1, parametrizer1, aggregator, args.cbm, args.senn, device) 
    elif args.model_name == 'dpl_auc':
        model = DPL_AUC(conceptizer1, parametrizer1, aggregator, args.cbm, args.senn, device)
    # send models to device you want to use
    model = model.to(device)
    
    # training all models. This function is in trainers.py    
    trainer = GradPenaltyTrainer(model, args, device)

    # train
    trainer.train(train_loader, valid_loader, epochs = args.epochs, save_path = model_path)
    
    # make figures
    trainer.plot_losses(save_path=results_path)

    # evaluation by test dataset
    trainer.test_and_save(test_loader,  save_file_name,       fold = 'test')
    trainer.test_and_save(train_loader, save_file_name_train, fold = 'train')

    # send model result to cpu
    model.eval().to("cpu")
    
    if args.wandb is not None:
        wandb.finish()

    """
    This function is in SENN/utils.py (lines 591-). 
    This function makes figures "grad*/concept_grid.pdf", which represents the maximize and minimize each unit in the concept layer
    """
    #concept_grid(model, pretrained_model, test_loader, top_k = 10, device="cpu", save_path = results_path + '/concept_grid.pdf')

    
if __name__ == '__main__':
    args = parse_args()
    # the number of task class
    args.nclasses = 5
    args.theta_dim = args.nclasses

    print(args)
    
    main(args)
