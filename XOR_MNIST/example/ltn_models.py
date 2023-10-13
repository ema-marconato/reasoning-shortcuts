import torch
import numpy as np
import itertools
import torch.nn.functional as F
import ltn

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

class identity(torch.nn.Module):
    def __init__(self):
        super(identity,self).__init__()
    def forward(self,x,d):
        return torch.gather(x, 1, d)

class LTN(torch.nn.Module):
    def __init__(self, hidden_neurons,args):
        super(LTN, self).__init__()
        self.hidden = hidden_neurons
        
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

        self.iden = identity()
        self.ltn = ltn.Predicate(self.iden)

        self.device = 'cpu'
        
    def forward(self, x):
        if not self.args.disent:
            logitC = self.mlp(x)
        else:
            logitC = torch.cat([
                        self.layer1(x[:,0].view(-1,1)), 
                        self.layer2(x[:,1].view(-1,1)),
                        self.layer3(x[:,2].view(-1,1))
                               ], dim=1)
            
        logitC += 0.2*torch.randn_like(logitC)
        
        pC = torch.zeros((8,6), device=self.device)
        for i in range(3):
            pC[:, 2*i]   = (1 - logitC[:,i].sigmoid() + 1e-3) / (1+ 2*1e-3)
            pC[:, 2*i+1] = (logitC[:,i].sigmoid() + 1e-3)  / (1+ 2*1e-3)

        # explicit logical rule on argmax
        pred = ( pC[:,:2].argmax(dim=1) + pC[:,2:4].argmax(dim=1) + pC[:,4:].argmax(dim=1)) % 2
        # assert torch.sum(torch.round(pred, decimals=3))/len(pred) == 1, (pred,torch.sum(torch.round(pred, decimals=3))/len(pred)) 
        # pred = torch.einsum('ki,kj,kl, mijl->km', pC[:,:2], pC[:,2:4], pC[:,4:], self.W)

        return pred, logitC, pC

class LTN_R(torch.nn.Module):
    def __init__(self, hidden_neurons, args):
        super(LTN_R, self).__init__()
        self.hidden = hidden_neurons
        self.encoder = LTN(hidden_neurons, args)
        self.decoder = MLPRecon(hidden_neurons)
        self.device = 'cpu'
        self.temp = 2
    def forward(self, x):
        pred, logitC, pC = self.encoder(x)
        recon = self.decoder(logitC)
        return pred, recon, logitC, pC

    def update_temp(self):
        self.temp = self.temp*0.95