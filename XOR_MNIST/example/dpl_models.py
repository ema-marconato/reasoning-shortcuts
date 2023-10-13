import torch
import numpy as np
import itertools
import torch.nn.functional as F

def create_world_matrix():
    W = torch.zeros((2,2,2,2), dtype=torch.float)
    for i, j, k in itertools.product(range(2), range(2), range(2)):
        W[:,i,j,k] = torch.tensor([1,0]) if (i+j+k % 2 == 0) else torch.tensor([0,1])
    return W

def create_uk():
    W = torch.zeros(2,2,2)
    for i, j, k in itertools.product(range(2), range(2), range(2)):
        W[i,j,k] = 1 if ( (i+j+k+1) % 2 == 0) else 0
    return W

class MLPRecon(torch.nn.Module):
    def __init__(self, hidden_neurons):
        super(MLPRecon, self).__init__()
        self.hidden = hidden_neurons
        
        self.layer1 = torch.nn.Linear(3, self.hidden, dtype=torch.float)
        self.layer2 = torch.nn.Linear(self.hidden, self.hidden, dtype=torch.float)
        self.layer3 = torch.nn.Linear(self.hidden, 3, dtype=torch.float)
        
    def forward(self, x):
        x = torch.nn.ReLU()(self.layer1(x))
        x = torch.nn.ReLU()(self.layer2(x))
        pG = torch.sigmoid(self.layer3(x) )

        pG = pG + 1e-3
        pG = pG / (1+3*1e-3)
        return pG
    
####### 

class DPL(torch.nn.Module):
    def __init__(self, hidden_neurons, args):
        super(DPL, self).__init__()
        
        if not args.disent:
            self.hidden = hidden_neurons
            
            self.layer1 = torch.nn.Linear(3, self.hidden, dtype=torch.float)
            self.layer2 = torch.nn.Linear(self.hidden, self.hidden, dtype=torch.float)
            self.layer3 = torch.nn.Linear(self.hidden, 3, dtype=torch.float)

            self.mlp = torch.nn.Sequential(self.layer1, torch.nn.ReLU(),
                                        self.layer2, torch.nn.ReLU(),
                                        self.layer3
                                        )
        elif not args.s_w:
            self.layer1 = torch.nn.Linear(1, 1, dtype=torch.float)
            self.layer2 = torch.nn.Linear(1, 1, dtype=torch.float)
            self.layer3 = torch.nn.Linear(1, 1, dtype=torch.float)

        else:
            self.layer1 = torch.nn.Linear(1, 1, dtype=torch.float)
            self.layer2 = self.layer1
            self.layer3 = self.layer1

        self.args = args

        self.device = 'cpu'
        
        self.W = create_uk().to(self.device)
    def forward(self, x):

        if not self.args.disent:
            logitC = self.mlp(x)
        else:
            logitC = torch.cat([
                        self.layer1(x[:,0].view(-1,1)), 
                        self.layer2(x[:,1].view(-1,1)),
                        self.layer3(x[:,2].view(-1,1))
                               ], dim=1)
        
        pC = torch.zeros((8,6), device=self.device)
        for i in range(3):
            pC[:, 2*i]   = (1 - logitC[:,i].sigmoid() + 1e-5) / (1+ 2*1e-5)
            pC[:, 2*i+1] = (logitC[:,i].sigmoid() + 1e-5)  / (1+ 2*1e-5)

        pred = torch.zeros((8,2), device=self.device)
        for i,j,k in itertools.product(range(2),range(2),range(2)):
            pred[:,1] += pC[:,i]*pC[:,2+j]*pC[:,4+k]*self.W[i,j,k]
        pred[:,0] = 1 - pred[:,1]
        pred = (pred + 1e-3)/(1+2*1e-3)
        # print(pC, pred)
        # quit()
        # assert torch.sum(torch.round(pred, decimals=3))/len(pred) == 1, (pred,torch.sum(torch.round(pred, decimals=3))/len(pred)) 
        # pred = torch.einsum('ki,kj,kl, mijl->km', pC[:,:2], pC[:,2:4], pC[:,4:], self.W)
        return pred, logitC

class DPL_R(torch.nn.Module):
    def __init__(self, hidden_neurons, args):
        super(DPL_R, self).__init__()
        self.hidden = hidden_neurons
        self.encoder = DPL(hidden_neurons, args)
        self.decoder = MLPRecon(hidden_neurons=6)
        self.device = 'cpu'
        self.temp = 2
    def forward(self, x):
        pred, logitC = self.encoder(x)
            
        C = torch.zeros((8,6), device=self.device)
        for i in range(3):
            C[:, 2*i]   = logitC[:,i] / 2
            C[:, 2*i+1] = -logitC[:,i] / 2
        C[:,:2]  = F.gumbel_softmax(C[:,:2],  tau=1, hard=True, dim=1)
        C[:,2:4] = F.gumbel_softmax(C[:,2:4], tau=1, hard=True, dim=1)
        C[:,4:]  = F.gumbel_softmax(C[:,4:],  tau=1, hard=True, dim=1)

        recon = self.decoder(C[:, [1,3,5]])
        return pred, recon, logitC

    def update_temp(self):
        self.temp = self.temp*0.95