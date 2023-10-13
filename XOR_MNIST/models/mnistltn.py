import torch
from utils.args import *
from utils.conf import get_device
from utils.losses import *
from models.cext import CExt
from models.utils.utils_problog import build_worlds_queries_matrix
from utils.ltn_loss import ADDMNIST_SAT_AGG

def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Learning via'
                                        'Concept Extractor .')
    add_management_args(parser)
    add_experiment_args(parser)
    return parser

class MnistLTN(CExt):
    NAME = 'mnistltn'
    '''
    MNIST OPERATIONS AMONG TWO DIGITS. IT WORKS ONLY IN THIS CONFIGURATION.
    '''
    def __init__(self, encoder, n_images=2, c_split=(), args=None):
        super(MnistLTN, self).__init__(encoder=encoder, 
                                      n_images=n_images, c_split=c_split)
        
        self.p = 2

        if args.task == 'addition':
            self.n_facts = 10 if not args.dataset == 'restrictedmnist' else 5
        elif args.task == 'product':    
            self.n_facts = 10 if not args.dataset == 'restrictedmnist' else 5
        elif args.task == 'multiop':    
            self.n_facts = 10 if not args.dataset == 'restrictedmnist' else 5

        # opt and device
        self.task = args.task
        self.opt = None
        self.device = get_device()

    def forward(self, x):
        # Image encoding
        cs = []
        xs = torch.split(x, x.size(-1) // self.n_images, dim=-1)
        for i in range(self.n_images):
            lc, _, _  = self.encoder(xs[i]) # sizes are ok
            cs.append(lc)
        clen = len(cs[0].shape)        
        cs = torch.stack(cs, dim=1) if clen == 2 else torch.cat(cs, dim=1)

        # normalize concept preditions
        pCs = self.normalize_concepts(cs)
        
        if self.task == 'addition':
            pred = torch.argmax(pCs[:,0,:], dim=-1) + torch.argmax(pCs[:,1,:],dim=-1)
            pred = F.one_hot(pred, 19)
        elif self.task == 'product':    
            pred = torch.argmax(pCs[:,0,:], dim=-1) * torch.argmax(pCs[:,1,:],dim=-1)
            pred = F.one_hot(pred, 82)
        elif self.task == 'multiop':    
            pred = torch.argmax(pCs[:,0,:], dim=-1)**2 + \
                   torch.argmax(pCs[:,1,:],dim=-1)**2  + \
                   torch.argmax(pCs[:,0,:], dim=-1) * torch.argmax(pCs[:,1,:],dim=-1) 
            mask = pred > 14
            pred[mask] = torch.tensor(15, device=pred.device)
            pred = F.one_hot(pred, 16)
        return {'CS': cs, 'YS':pred, 'pCS': pCs}

    def normalize_concepts(self, z, split=2):
        """Computes the probability for each ProbLog fact given the latent vector z"""
        # Extract probs for each digit

        prob_digit1, prob_digit2 = z[:, 0,:], z[:, 1,:]

        # Sotfmax on digits_probs (the 10 digits values are mutually exclusive)
        prob_digit1 = torch.nn.Softmax(dim=1)(prob_digit1)
        prob_digit2 = torch.nn.Softmax(dim=1)(prob_digit2)

        # Clamp digits_probs to avoid underflow
        eps = 1e-3
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
            return ADDMNIST_SAT_AGG(ADDMNIST_Cumulative, self.task)
        else: return NotImplementedError('Wrong dataset choice')
        
    def start_optim(self, args):
        self.opt = torch.optim.Adam(self.parameters(), args.lr)