import itertools
import copy
from exp_best_args import *


def launch_XOR(args):
    # define setting
    args.project="XOR"
    args.wandb = True
    
    args.seed = 0
    
    hyperparameters=[
            ['dpl', 'sl', 'ltn'],  # model
            [True, False], # rec
            [True, False], # entropy
            [0.1, 1, 10], # gamma
            [1e-4, 5*1e-4, 1e-3, 5*1e-3, 1e-2] # lr
        ]

    args_list=[]
    for element in itertools.product(*hyperparameters):
        args1= copy.copy(args)
        args1.model, args1.rec, args1.entropy, args.gamma, args.lr = element
        print(args1, '\n')
        args_list.append(args1)
    return args_list

def launch_XOR_exp1(args):
    # define setting
    args.project="XOR-paper"
    args.wandb = True
        
    hyperparameters=[
            ['ltn'],  # model
            [True], # disent
            [False], # shared weights
            [i for i in range(30, 40)], # lr
            [1e-2, 0.05]
        ]

    args_list=[]
    for element in itertools.product(*hyperparameters):
        args1= copy.copy(args)
        args1.model, args1.disent, args1.s_w, args1.seed, args1.lr = element
        
        if args1.disent or not args1.s_w:            
            args1 = set_best_args_XOR(args1)
            
            print(args1, '\n')
            
            args_list.append(args1)
        
    return args_list


def launch_XOR_debug(args):
    # define setting
    args.project="XOR-debug"
    args.wandb = True
    
    args.seed = 0
    
    hyperparameters=[
            ['dpl'],  # model
            [True], # rec
        ]

    args_list=[]
    for element in itertools.product(*hyperparameters):
        args1= copy.copy(args)
        args1.model, args1.rec = element
        print(args1, '\n')
        args_list.append(args1)
    return args_list
    

def launch_addmnist(args):
    '''
    Experiments for Addition-MNIST sequential for nesy
    '''
    # define setting
    args.dataset="addmnist"
    args.n_epochs=50
    args.c_sup=0
    args.batch_size=256

    # set project
    args.project="addmnist"
    args.validate = False
    args.wandb = 'name'
    
    args.lr = 1e-3

    hyperparameters=[
            ['mnistltn'],  # model
            [True],  # joint
            [False],  # splitted 
            [i for i in range(50,100)] # lr
        ]

    args_list=[]
    for element in itertools.product(*hyperparameters):
        args1= copy.copy(args)
        args1.model, args1.joint, args1.splitted, args1.seed = element
        
        print(args1, '\n')        
        args_list.append(args1)
            
    return args_list


def launch_sl_gridsearch(args):
    '''
    Experiments for Addition-MNIST sequential for nesy
    '''
    # define setting
    args.dataset="addmnist"
    args.n_epochs=25
    args.c_sup=0
    args.batch_size=256

    # set project
    args.project="addmnist-sl"
    args.validate = True
    args.wandb = 'name'
    
    args.seed = 0

    hyperparameters=[
            ['mnistsl'],  # model
            [0.5,1,2,5,10],  # weight of SL
            [1e-4, 5*1e-4, 1e-3, 5*1e-3] # lr
        ]

    args_list=[]
    for element in itertools.product(*hyperparameters):
        args1= copy.copy(args)
        args1.model, args1.w_sl, args1.lr = element
        print(args1, '\n')
        args_list.append(args1)
    return args_list

def launch_short_search(args):
    '''
    Experiments for Addition-MNIST sequential for nesy
    '''
    # define setting
    args.dataset="shortmnist"
    args.n_epochs=50
    args.warmup_steps = 5
    args.batch_size=256
    
    args.exp_decay = 0.95

    # set project
    args.project="shortcut-weighted-gridsearch"
    args.validate = True
    args.wandb = 'name'
    args.joint = False
    
    args.which_c = [4,9]
    args.w_c = 1
    
    args.seed = 0
    args.lr = 5*1e-3

    hyperparameters=[
            ['mnistltn'],  # model
            # ['mnistltnrec'], 
            [0],  #csup
            [False], #entropy
            [1e-4, 1e-3, 1e-2] # gamma
        ]

    args_list=[]
    for element in itertools.product(*hyperparameters):
        args1 = copy.copy(args)
        args1.model, args1.c_sup, args.entropy, args1.lr = element
        
        # args1 = set_best_args_shortmnist(args1)
           
        # case1 = (args1.model == 'mnistdpl' and args1.c_sup == 1 and not args1.entropy)
        # case2 = (args1.model == 'mnistdplrec')
        # case3 = (args1.model == 'mnistltn' and args1.c_sup == 1)
        # case4 = (args1.model == 'mnistltnrec')
        
        # if case1 or case2 or case3 or case4:        
        args_list.append(args1)
        print(args1)
        
    return args_list

def launch_joint(args):
    args.dataset="addmnist"
    args.n_epochs=30
    args.warmup_steps = 5
    args.c_sup=0
    args.batch_size=256

    # set project
    args.project="joint-gridsearch"
    args.validate = True
    args.wandb = 'name'
    args.joint = True
    
    args.seed = 0
    
    #mitigations
    args.entropy = True
    
    hyperparameters=[
            ['mnistdpl', 'mnistsl', 'mnistltn'],  # model
            [0.1, 0.5, 1, 2, 5, 10], #mitigation
            [1e-4, 5*1e-4, 1e-3, 5*1e-3] # lr
        ]

    args_list=[]
    for element in itertools.product(*hyperparameters):
        args1= copy.copy(args)
        args1.model, args1.gamma, args1.lr = element
        print(args1, '\n')
        args_list.append(args1)
    return args_list


def launch_multiop(args):
    '''
    Experiments for Addition-MNIST sequential for nesy
    '''
    # define setting
    args.dataset="restrictedmnist"
    args.n_epochs=50
    args.c_sup=0
    args.batch_size=256

    # set project
    args.project="multiop"
    args.validate = False
    args.wandb = 'name'
    
    args.lr = 1e-4

    hyperparameters=[
            ['mnistltn'],  # model
            ['multiop'],
            [i for i in range(10)] # seed
        ]

    args_list=[]
    for element in itertools.product(*hyperparameters):
        args1= copy.copy(args)
        args1.model, args1.task, args1.seed = element
        
        print(args1, '\n')        
        args_list.append(args1)
            
    return args_list
