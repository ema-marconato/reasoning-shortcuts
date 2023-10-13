import itertools
import copy

def launch_bdd(args):
    '''
    Experiments for BDD-OIA for nesy
    '''
    
    # default args
    args.train=True
    args.cuda=False
    args.h_type = 'fcc'
    
    # set project
    args.project="BOIA"
    args.wandb = 'yours'    
    
    # standard hypes
    args.epochs = 10
    args.batch_size = 512
    args.nconcepts = 30
    args.nconcepts_labeled = 21
    args.h_sparsity = 7
    args.opt = 'adam'
    args.lr = 0.005
    args.weight_decay = 0.00004 
    args.theta_reg_lambda = 0.001
    args.obj = 'bce'
    
    # which concepts
    args.which_c = [-1] # [0,1,2,3,4,5,6,7,8]
        
    args.seed = 0
    
    # hyperparameters=[
    #         ['dpl_auc'], 
    #         [0],  #csup
    #         [0], # entropy
    #         [i for i in range(10)] # seed
    #     ]
    
    hyperparameters=[
            ['dpl_auc'], 
            [0,1],  #csup
            [0,1], # entropy
            [i for i in range(10)] # seed
        ]

    args_list=[]
    for element in itertools.product(*hyperparameters):
        args1 = copy.copy(args)
        
        args1.model_name, args1.h_labeled_param, args1.w_entropy, args1.seed = element        
        
        
        if args1.model_name == 'dpl_auc':
            args1.h_labeled_param = 0.01*args1.h_labeled_param    
        
        
        print(args1)
        print()
        
        args_list.append(args1)
        
    return args_list
