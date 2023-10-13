import torch
import torch.nn.functional as F
from models.utils.deepproblog_modules import DeepProblogModel
from utils.losses import ADDMNIST_rec_class
from utils.args import *
from utils.conf import get_device

def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Variational-Conceptual Autoencoders.')
    add_management_args(parser)
    add_experiment_args(parser)
    return parser

class CVAE(torch.nn.Module):
    NAME = 'cvae'
    def __init__(self, encoder= None, decoder=None,
                 n_images=1, c_split=()):
        super(CVAE,self).__init__()
        
        # bones of model
        self.encoder = encoder
        self.decoder = decoder
        
        # number of images, and how to split them in cat concepts
        self.n_images = n_images
        self.c_split = c_split 

        # opt and device
        self.opt = None
        self.device = get_device()

    def forward(self, x):
        # extract encodings
        cs, mus, logvars = [], [], []
        recs = []
        xs = torch.split( x, x.size(-1)//self.n_images, -1)
        for i in range(self.n_images):
            c, mu, logvar  = self.encoder(xs[i])
            cs.append(c)
            mus.append(mu)
            logvars.append(logvar)
    
            #extract decodings

            # 1) convert to variational
            eps = torch.randn_like(logvar)
            zs = mu + eps*logvar.exp()

            index, hard_cs = 0, []
            for cdim in self.c_split:
                hard_cs.append( F.gumbel_softmax(c[:, index:index+cdim] , tau=1, hard=True, dim=-1) )
                index += cdim
            hard_cs = torch.cat(hard_cs, dim=-1)

            # 2) pass to decoder
            decode = self.decoder(torch.cat([hard_cs, zs], dim=-1))
            recs.append(decode)
        
        # return everything
        cs = torch.cat(cs, dim=-1)
        mus = torch.cat(mus, dim=-1)
        logvars = torch.cat(logvars, dim=-1)
        recs = torch.cat(recs, dim=-1)
        return {'RECS': recs, 'CS': cs, 'MUS': mus, 'LOGVARS': logvars}

    @staticmethod
    def get_loss(args):
        if args.dataset == 'addmnist':
            return ADDMNIST_rec_class
        else: return NotImplementedError('Wrong dataset choice')
        
    def start_optim(self, args):
        self.opt = torch.optim.Adam(self.parameters(), args.lr)