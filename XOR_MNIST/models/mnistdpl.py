import torch
from models.utils.deepproblog_modules import DeepProblogModel
from utils.args import *
from utils.conf import get_device
from models.utils.deepproblog_modules import GraphSemiring
from models.utils.utils_problog import *
from utils.losses import *
from utils.dpl_loss import ADDMNIST_DPL


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Learning via'
                                        'Concept Extractor .')
    add_management_args(parser)
    add_experiment_args(parser)
    return parser

class MnistDPL(DeepProblogModel):
    NAME = 'mnistdpl'
    '''
    MNIST OPERATIONS AMONG TWO DIGITS. IT WORKS ONLY IN THIS CONFIGURATION.
    '''
    def __init__(self, encoder, n_images=2,
                 c_split=(), args=None,
                 model_dict=None,
                 n_facts=20, nr_classes=19):
        super(MnistDPL, self).__init__(encoder=encoder, model_dict=model_dict,
                                       n_facts=n_facts, nr_classes=nr_classes)
        
        # how many images and explicit split of concepts
        self.n_images = n_images
        self.c_split = c_split

        # Worlds-queries matrix
        if args.task == 'addition':
            self.n_facts = 10 if not args.dataset == 'restrictedmnist' else 5
            self.w_q = build_worlds_queries_matrix(2, self.n_facts, 'addmnist')
            self.nr_classes = 19
        elif args.task == 'product':
            self.n_facts = 10 if not args.dataset == 'restrictedmnist' else 5
            self.w_q = build_worlds_queries_matrix(2, self.n_facts, 'productmnist')
            self.nr_classes = 37
        elif args.task == 'multiop':
            self.n_facts = 5
            self.w_q = build_worlds_queries_matrix(2, self.n_facts, 'multiopmnist')
            self.nr_classes = 3

        # opt and device
        self.opt = None
        self.device = get_device()
        self.w_q = self.w_q.to(self.device)

    def forward(self, x):
        # Image encoding
        cs = []
        xs = torch.split(x, x.size(-1) // self.n_images, dim=-1)
        for i in range(self.n_images):
            lc, mu, logvar = self.encoder(xs[i]) # sizes are ok
            cs.append(lc)
        clen = len(cs[0].shape)        
        cs = torch.stack(cs, dim=1) if clen == 2 else torch.cat(cs, dim=1)
        
        # normalize concept preditions
        pCs = self.normalize_concepts(cs)
        
        # Problog inference to compute worlds and query probability distributions
        py, worlds_prob = self.problog_inference(pCs)
        
        return {'CS': cs, 'YS':py, 'pCS': pCs}
    
    def problog_inference(self, pCs, query=None):
        """
        Performs ProbLog inference to retrieve the worlds probability distribution P(w).
        Works with two encoded bits.
        """
        
        # Extract first and second digit probability
        prob_digit1, prob_digit2 = pCs[:, 0, :], pCs[:, 1,:]

        # Compute worlds probability P(w) (the two digits values are independent)
        Z_1 = prob_digit1[..., None]
        Z_2 = prob_digit2[:, None, :]

        # print(Z_1.size())
        # print(Z_2.size())

        probs = Z_1.multiply(Z_2)

        # print(probs.size())

        worlds_prob = probs.reshape(-1, self.n_facts*self.n_facts)

        # Compute query probability P(q)
        query_prob = torch.zeros(size=(len(probs), self.nr_classes), device=probs.device)

        for i in range(self.nr_classes):
            query = i
            query_prob[:,i] = self.compute_query(query, worlds_prob).view(-1)

        # add a small offset
        query_prob += 1e-5
        with torch.no_grad():
            Z = torch.sum(query_prob, dim=-1, keepdim=True)
        query_prob = query_prob / Z

        return query_prob, worlds_prob

    def compute_query(self, query, worlds_prob):
            """Computes query probability given the worlds probability P(w)."""
            # Select the column of w_q matrix corresponding to the current query
            w_q = self.w_q[:, query]
            # Compute query probability by summing the probability of all the worlds where the query is true
            query_prob = torch.sum(w_q * worlds_prob, dim=1, keepdim=True)
            return query_prob

    def normalize_concepts(self, z, split=2):
        """Computes the probability for each ProbLog fact given the latent vector z"""
        # Extract probs for each digit

        prob_digit1, prob_digit2 = z[:, 0,:], z[:, 1,:]

        # # Add stochasticity on prediction
        # prob_digit1 += 0.5 * torch.randn_like(prob_digit1, device=prob_digit1.device)
        # prob_digit2 += 0.5 * torch.randn_like(prob_digit2, device=prob_digit1.device)

        # Sotfmax on digits_probs (the 10 digits values are mutually exclusive)
        prob_digit1 = nn.Softmax(dim=1)(prob_digit1)
        prob_digit2 = nn.Softmax(dim=1)(prob_digit2)

        # Clamp digits_probs to avoid ProbLog underflow
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
    
    @staticmethod
    def get_loss(args):
        if args.dataset in ['addmnist', 'shortmnist', 'restrictedmnist']:
            return ADDMNIST_DPL(ADDMNIST_Cumulative)
        else: return NotImplementedError('Wrong dataset choice')
        
    def start_optim(self, args):
        self.opt = torch.optim.Adam(self.parameters(), args.lr)