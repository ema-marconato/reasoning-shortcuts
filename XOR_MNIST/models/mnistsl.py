import torch
from utils.args import *
from utils.conf import get_device
from utils.losses import *
from models.cext import CExt
from models.utils.utils_problog import build_worlds_queries_matrix
from utils.semantic_loss import ADDMNIST_SL

def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Learning via'
                                        'Concept Extractor .')
    add_management_args(parser)
    add_experiment_args(parser)
    return parser

class MnistSL(CExt):
    NAME = 'mnistsl'
    '''
    MNIST OPERATIONS AMONG TWO DIGITS. IT WORKS ONLY IN THIS CONFIGURATION.
    '''
    def __init__(self, encoder, n_images=2,
                 c_split=(), args=None,
                 n_facts=20, nr_classes=19):
        super(MnistSL, self).__init__(encoder=encoder, 
                                      n_images=n_images, c_split=c_split)
        
        #  Worlds-queries matrix
        if args.task == 'addition':
            self.n_facts = 10 if not args.dataset == 'restrictedmnist' else 5
            self.logic = build_worlds_queries_matrix(2, self.n_facts, 'addmnist')
            self.nr_classes = 19
        elif args.task == 'product':
            self.n_facts = 10 if not args.dataset == 'restrictedmnist' else 5
            self.logic = build_worlds_queries_matrix(2, self.n_facts, 'productmnist')
            self.nr_classes = 37
        elif args.task == 'multiop':
            self.n_facts = 5
            self.logic = build_worlds_queries_matrix(2, self.n_facts, 'multiopmnist')
            self.nr_classes = 3

        self.mlp = torch.nn.Sequential(torch.nn.Linear(self.n_facts*2, 50),
                                       torch.nn.ReLU(),
                                       torch.nn.Linear(50, 50),
                                       torch.nn.ReLU(),
                                       torch.nn.Linear(50, self.nr_classes))
        
        # opt and device
        self.opt = None
        self.device = get_device()
        self.logic  = self.logic.to(self.device)

    def forward(self, x):
        # Image encoding
        cs = []
        xs = torch.split(x, x.size(-1) // self.n_images, dim=-1)
        for i in range(self.n_images):
            lc, _, _  = self.encoder(xs[i]) # sizes are ok
            cs.append(lc)
        clen = len(cs[0].shape)        
        cs = torch.stack(cs, dim=1) if clen == 2 else torch.cat(cs, dim=1)

        pCs = self.normalize_concepts(cs)
        
        # normalize concept preditions
        
        pred = self.mlp(cs.view(-1,self.n_facts*2))
        
        return {'CS': cs, 'YS':pred, 'pCS': pCs}

    def normalize_concepts(self, z, split=2):
        """Computes the probability for each ProbLog fact given the latent vector z"""
        # Extract probs for each digit

        prob_digit1, prob_digit2 = z[:, 0,:], z[:, 1,:]

        # Sotfmax on digits_probs (the 10 digits values are mutually exclusive)
        prob_digit1 = torch.nn.Softmax(dim=1)(prob_digit1)
        prob_digit2 = torch.nn.Softmax(dim=1)(prob_digit2)

        # Clamp digits_probs to avoid underflow
        eps = 1e-5
        prob_digit1 = prob_digit1 + eps
        with torch.no_grad():
            Z1 = torch.sum(prob_digit1, dim=-1, keepdim=True)
        prob_digit1 = prob_digit1 / Z1  # Normalization
        prob_digit2 = prob_digit2 + eps
        with torch.no_grad():
            Z2 = torch.sum(prob_digit2, dim=-1, keepdim=True)
        prob_digit2 = prob_digit2 / Z2  # Normalization

        return torch.stack([prob_digit1, prob_digit2], dim=1).view(-1, 2, self.n_facts)
    
    def get_loss(self, args):
        if args.dataset in ['addmnist', 'shortmnist', 'restrictedmnist']:
            return ADDMNIST_SL(ADDMNIST_Cumulative, self.logic, args)
        else: return NotImplementedError('Wrong dataset choice')
        
    def start_optim(self, args):
        self.opt = torch.optim.Adam(self.parameters(), args.lr)