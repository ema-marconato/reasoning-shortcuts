import torch
from models.utils.deepproblog_modules import DeepProblogModel
from utils.losses import ADDMNIST_Concept_Match
from utils.args import *
from utils.conf import get_device


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Learning via'
                                        'Concept Extractor .')
    add_management_args(parser)
    add_experiment_args(parser)
    return parser

class CExt(torch.nn.Module):
    NAME = 'cext'
    def __init__(self, encoder,
                 n_images=1, c_split=()): # c_dim=20, latent_dim=0):
        super(CExt, self).__init__()
        
        # bones of the model 
        self.encoder = encoder

        # number of images, and how to split them
        self.n_images = n_images
        self.c_split = c_split
        # self.c_dim = c_dim
        # self.latent_dim =latent_dim

        # opt and device
        self.opt = None
        self.device = get_device()

    def forward(self, x):
        cs = []
        xs = torch.split(x, x.size(-1) // self.n_images, dim=-1)
        for i in range(self.n_images):
            cs.append( self.encoder(xs[i])[0] )
        return {'CS': torch.stack(cs, dim=1)}

    @staticmethod
    def get_loss(args):
        if args.dataset in ['addmnist', 'shortmnist']:
            return ADDMNIST_Concept_Match
        else: return NotImplementedError('Wrong Choice')
        
    def start_optim(self, args):
        self.opt = torch.optim.Adam(self.parameters(), args.lr)