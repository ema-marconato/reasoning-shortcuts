def set_best_args_shortmnist(args1):
    args1.gamma = 5
    
    C = (args1.c_sup > 0)
    H =  args1.entropy
    R = (args1.model in ['mnistdlrec', 'mnistslrec', 'mnistltnrec'])
    
    print(R, H, C)
    
    if args1.model in ['mnistdpl', 'mnistdplrec']:
        args1.beta  = 2
        args1.w_rec = 1
        args1.w_h = 0.5
        args1.w_c = 1
        
        # single term
        if R and not H and not C: args1.lr = 5*1e-3; args1.gamma =1
        if H and not R and not C: args1.lr = 5*1e-4; args1.gamma =1
        if C and not R and not H: args1.lr = 5*1e-3; args1.gamma=0.1
        
        # two terms
        if H and C and not R:     args1.lr = 1e-3;   args1.gamma = 2
        if H and R and not C:     args1.lr = 5*1e-3; args1.gamma = 1
        if R and C and not H:     args1.lr = 1e-3;   args1.gamma = 0.1
        
        # all terms
        if H and C and R:         args1.lr=5*1e-3;   args1.gamma = 1 
        
    
    if args1.model in ['mnistsl', 'mnistslrec']:
        args1.beta  = 0.1
        args1.w_rec = 0.1
        args1.w_h = 2
        args1.w_c = 5
        
        # single term
        if R and not H and not C: args1.lr = 5*1e-3; args1.gamma =1
        if H and not R and not C: args1.lr = 5*1e-3; args1.gamma =1
        if C and not R and not H: args1.lr = 5*1e-3; args1.gamma =1
        
        # two terms
        if H and C and not R:     args1.lr = 5*1e-3; args1.gamma = 2
        if H and R and not C:     args1.lr = 5*1e-3; args1.gamma = 2
        if R and C and not H:     args1.lr = 5*1e-3; args1.gamma = 2
        
        # all terms
        if H and C and R:         args1.lr=5*1e-3;   args1.gamma = 2

    if args1.model in ['mnistltn', 'mnistltnrec']:
        args1.beta  = 0.1
        args1.w_rec = 0.5
        args1.w_h = 2
        args1.w_c = 0.01
        
        # single term
        if R and not H and not C: args1.lr = 5*1e-3; args1.gamma =1
        if H and not R and not C: args1.lr = 5*1e-4; args1.gamma =1
        if C and not R and not H: args1.lr = 1e-3;   args1.gamma =1
        
        # two terms
        if H and C and not R:     args1.lr = 5*1e-3; args1.gamma = 2
        if H and R and not C:     args1.lr = 5*1e-3; args1.gamma = 0.1
        if R and C and not H:     args1.lr = 5*1e-3; args1.gamma = 0.1
        
        # all terms
        if H and C and R:         args1.lr=5*1e-3;   args1.gamma = 2
    
    return args1

def set_best_args_XOR(args1):
    
    if args1.model == 'dpl':
        if not args1.disent: args1.lr = 0.05
        elif args1.disent:   args1.lr = 0.01
        if args1.s_w:        args1.lr = 0.005
    
    if args1.model == 'sl':
        if not args1.disent: args1.lr = 0.01
        elif args1.disent:   args1.lr = 0.01
        if args1.s_w:        args1.lr = 0.01

    if args1.model == 'ltn':
        if not args1.disent: args1.lr = 0.01
        elif args1.disent:   args1.lr = 0.01
        if args1.s_w:        args1.lr = 0.01
    
    return args1
