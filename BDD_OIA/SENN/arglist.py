# -*- coding: utf-8 -*-
"""
list of hyperparameters
Example is in parameter.txt so please see it
"""

import pdb
import argparse
import torch

def tensor_to_numpy(tensor):
    return tensor.data[0]

class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def get_senn_parser():
    parser = argparse.ArgumentParser(description='Self-Explaining Neural Net Classifier')

    ### Overall Setup
    parser.add_argument('--train', action='store_true', default=False, help='Whether or not to train model')
    parser.add_argument('--test', action='store_true', default=False, help='Whether or not to run model on test set')
    parser.add_argument('--load_model', action='store_true', default=False, help='Load pretrained model from default path')

    ### Save Paths
    parser.add_argument('--model_path', type=str, default='models', help='where to save the snapshot')
    parser.add_argument('--results_path', type=str, default='out', help='where to dump model config and epoch stats')
    parser.add_argument('--log_path', type=str, default='log', help='where to dump training logs  epoch stats (and config??)')
    parser.add_argument('--summary_path', type=str, default='results/summary.csv', help='where to dump model config and epoch stats')

    ### Device
    parser.add_argument('--cuda', action='store_true', default=False, help='enable the gpu' )
    parser.add_argument('--num_gpus', type=int, default=2, help='Num GPUs to use.')
    parser.add_argument('--seed', type=int, default=2018, help='Set random seed.')


    ### Model

    # Concept Encoder (H)
    parser.add_argument('--cbm', default=True, help='type of conceptizer (learnt or input)' ) # newly added 
    parser.add_argument('--senn', action='store_true', default=False, help='type of conceptizer (learnt or input)' ) # newly added 
    parser.add_argument('--h_type', type=str, default='cnn', help='type of conceptizer (learnt or input)' ) # Don's change!!
    parser.add_argument('--concept_dim', type=int, default=1, help='concept dimension. dont change')
    parser.add_argument('--nconcepts_labeled', type=int, default=21, help='number of labeled concepts') # newly added
    parser.add_argument('--nconcepts', type=int, default=30, help='number of concepts')
    parser.add_argument('--h_sparsity', type=int, default=7, help='kWTA hyperparameter')

    # Added for weak supervision
    parser.add_argument('--make_label', action='store_true', default=False, help='making concepts labels or not.')
    parser.add_argument('--h_labeled_param', type=float, default=10, help='parameter for learning h [default: 1-e4]')
    parser.add_argument('--w_entropy', type=float, default=0, help='parameter for learning h [default: 1-e4]')
    parser.add_argument('--info_hypara', type=float, default=0.5, help='hyperparameter of info loss') #newly added

    # Parametrizing Function (Theta)
    parser.add_argument('--nobias', action='store_true', default=False, help='do not add a bias term theta_0' )
    parser.add_argument('--positive_theta', action='store_true', default=False, help="relevance scores in [0,1] instead of [-1,1]")
    parser.add_argument('--theta_arch', type=str, default='simple', help="Parametrizer architecture", choices= ['simple','alexnet', 'vgg8','vgg11_bn', 'vgg11', 'vgg16'])
    parser.add_argument('--theta_dim', type=int, default=-1, help="dimension of theta_i. deafults to number of classes")
    parser.add_argument('--theta_reg_type', type=str, default='grad3', help="Type of regularization on theta. [none|grad1-3|crosslip]")
    parser.add_argument('--theta_reg_lambda', type=float, default=1e-2, help="Stength of regularization on theta.")

    ### Learning
    parser.add_argument('--opt', type=str, default='adam', help='optim method [default: adam]')
    parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate [default: 0.001]')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs for train [default: 10]')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size for training [default: 64]')
    parser.add_argument('--objective', default='cross_entropy', help='choose which loss objective to use')
    parser.add_argument('--dropout', type=float, default=0.5, help='the probability for dropout [default: 0.5]')
    parser.add_argument('--weight_decay', type=float, default=1e-3, help='L2 norm penalty [default: 1e-3]')


    ### Data  --- FIXME: Not used yet. Maybe use to avoid duplication of main scripts for similar tasks (e.g. MNIST, CIFAR)
    parser.add_argument('--dataset', default='pathology', help='choose which dataset to run on')
    parser.add_argument('--embedding', default='pathology', help='choose what embeddings to use')
    parser.add_argument('--nclasses', type=int, default=2, help='number of classes' )

    ### Misc
    parser.add_argument('--num_workers' , type=int, default=30, help='num workers for data loader')
    parser.add_argument('--print_freq' , type=int, default=1, help='print frequency during train (in batches)')
    parser.add_argument('--debug', action='store_true', default=False, help='debug mode' )


    return parser


def parse_args():
    parser = argparse.ArgumentParser(description='Self-Explaining Neural Net Classifier')

    # setup
    parser.add_argument('--train', action='store_true', default=True, help='Whether or not to train model')
    parser.add_argument('--test', action='store_true', default=False, help='Whether or not to run model on test set')
    parser.add_argument('--load_model', action='store_true', default=False, help='Load pretrained model from default path')

    # device
    parser.add_argument('--cuda', action='store_true', default=False, help='enable the gpu' )
    parser.add_argument('--num_gpus', type=int, default=1, help='Num GPUs to use.')

    parser.add_argument('--debug', action='store_true', default=False, help='debug mode' )

    # learning
    parser.add_argument('--opt', type=str, default='adam', help='optim method [default: adam]')
    parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate [default: 0.001]')
    parser.add_argument('--epochs', type=int, default=3, help='number of epochs for train [default: 10]')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size for training [default: 64]')
    parser.add_argument('--objective', default='cross_entropy', help='choose which loss objective to use')

    #pathsn
    parser.add_argument('--model_path', type=str, default='models', help='where to save the snapshot')
    parser.add_argument('--results_path', type=str, default='out', help='where to dump model config and epoch stats')
    parser.add_argument('--log_path', type=str, default='log', help='where to dump training logs  epoch stats (and config??)')
    parser.add_argument('--summary_path', type=str, default='results/summary.csv', help='where to dump model config and epoch stats')


    # model
    parser.add_argument('--cbm', action='store_true', default=False, help='type of conceptizer (learnt or input)' ) # newly added
    parser.add_argument('--senn', action='store_true', default=False, help='type of conceptizer (learnt or input)' ) # newly added
    parser.add_argument('--h_type', type=str, default='cnn', help='type of conceptizer (learnt or input)' )
    #parser.add_argument('--learn_h', type='str', default='learnt', help='type of conceptizer (learnt or input)' )

    parser.add_argument('--concept_dim', type=int, default=1, help='concept dimension. dont change')
    parser.add_argument('--nconcepts_labeled', type=int, default=6, help='number of labeled concepts') # newly  added
    parser.add_argument('--nconcepts', type=int, default=20, help='number of concepts')
    parser.add_argument('--nobias', action='store_true', default=False, help='do not add a bias term theta_0' )

    parser.add_argument('--h_sparsity', type=int, default=7, help='kWTA hyperparameter')

    parser.add_argument('--positive_theta', action='store_true', default=False, help="relevance scores in [0,1] instead of [-1,1]")
    parser.add_argument('--theta_dim', type=int, default=-1, help="dimension of theta_i. deafults to number of classes")
    parser.add_argument('--theta_reg_type', type=str, default='unreg', help="Type of regularization on theta. [none|grad1-3|crosslip]")
    parser.add_argument('--theta_reg_lambda', type=float, default=1e-2, help="Stength of regularization on theta.")
    parser.add_argument('--dropout', type=float, default=0.1, help='the probability for dropout [default: 0.5]')
    parser.add_argument('--weight_decay', type=float, default=1e-3, help='L2 norm penalty [default: 1e-3]')

    # data
    parser.add_argument('--dataset', default='pathology', help='choose which dataset to run on')
    parser.add_argument('--embedding', default='pathology', help='choose what embeddings to use')
    parser.add_argument('--nclasses', type=int, default=2, help='number of classes' )

    # data loading
    parser.add_argument('--num_workers' , type=int, default=4, help='num workers for data loader')

    # misc
    parser.add_argument('--print_freq' , type=int, default=1, help='print frequency during train (in batches)')


    args = parser.parse_args()


    print("\nParameters:")
    for attr, value in sorted(args.__dict__.items()):
        print("\t{}={}".format(attr.upper(), value))

    return args
