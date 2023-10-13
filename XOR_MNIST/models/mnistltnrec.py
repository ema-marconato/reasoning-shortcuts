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

class MnistLTNRec(CExt):
    NAME = 'mnistltnrec'
    '''
    MNIST OPERATIONS AMONG TWO DIGITS. IT WORKS ONLY IN THIS CONFIGURATION.
    '''
    def __init__(self, encoder, decoder, n_images=2, c_split=(), args=None):
        super(MnistLTNRec, self).__init__(encoder=encoder, 
                                          n_images=n_images, 
                                          c_split=c_split)
        
        self.p = 2
        self.decoder = decoder

        # opt and device
        self.opt = None
        self.device = get_device()

    def forward(self, x):
        # Image encoding
        cs, mus, logvars = [], [], []
        latents = []
        recs = []
        xs = torch.split( x, x.size(-1)//self.n_images, -1)
        for i in range(self.n_images):
            c, mu, logvar  = self.encoder(xs[i])
            cs.append(c)
            mus.append(mu)
            logvars.append(logvar)
    
            #extract decodings

            # 1) add variational vars + discrete
            eps = torch.randn_like(logvar)
            L = len(eps)
            latents.append((mu + eps*logvar.exp()).view(L, -1))

            for i in range(len(self.c_split)):
                latents.append( F.gumbel_softmax(c[:, i, :] , tau=1, hard=True, dim=-1) )
        
        latents = torch.cat(latents, dim=1)

        # 2) pass to decoder
        recs = self.decoder(latents)
        
        # return everything
        clen = len(cs[0].shape)        
        cs      = torch.stack(cs, dim=1) if clen == 2       else torch.cat(cs, dim=1)
        mus     = torch.stack(mus, dim=-1) if clen == 2     else torch.cat(mus, dim=1)
        logvars = torch.stack(logvars, dim=-1) if clen == 2 else torch.cat(logvars, dim=1)
        
        # normalize concept preditions
        pCs = self.normalize_concepts(cs)
        
        # normalize concept preditions
        
        pred = torch.argmax(pCs[:,0,:], dim=-1) + torch.argmax(pCs[:,1,:],dim=-1)
        
        return {'YS': F.one_hot(pred, 19), 'CS': cs, 'pCS': pCs, 'MUS': mus, 'LOGVARS': logvars, 'RECS': recs}

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

        return torch.stack([prob_digit1, prob_digit2], dim=1).view(-1, 2, 10)
    
    def get_loss(self, args):
        if args.dataset in ['addmnist', 'shortmnist']:
            return ADDMNIST_SAT_AGG(ADDMNIST_Cumulative)
        else: return NotImplementedError('Wrong dataset choice')
        
    def start_optim(self, args):
        self.opt = torch.optim.Adam(self.parameters(), args.lr)