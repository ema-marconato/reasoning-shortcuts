import torch
import torch.nn.functional as F

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
    

class SL(torch.nn.Module):
    def __init__(self, hidden_neurons, args):
        super(SL, self).__init__()
        self.hidden = hidden_neurons
        self.args = args
        
        if not args.disent:
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

        self.predictor = torch.nn.Sequential(torch.nn.Linear(3, self.hidden),
                                             torch.nn.Tanh(),
                                             torch.nn.Linear(self.hidden, self.hidden),
                                             torch.nn.Tanh(),
                                             torch.nn.Linear(self.hidden, 3))

        self.device = 'cpu'
        
        self.W = torch.nn.Parameter(torch.rand(size=(2,2,2,2))).to(self.device)
    def forward(self, x):
        if not self.args.disent:
            logitC = self.mlp(x)
        else:
            logitC = torch.cat([
                        self.layer1(x[:,0].view(-1,1)), 
                        self.layer2(x[:,1].view(-1,1)),
                        self.layer3(x[:,2].view(-1,1))
                               ], dim=1)

        # get probs
        pC = torch.zeros((8,6), device=self.device)
        for i in range(3):
            pC[:, 2*i]   = (1 - logitC[:,i].sigmoid() + 1e-5) / (1+ 2*1e-5)
            pC[:, 2*i+1] = (logitC[:,i].sigmoid() + 1e-5)  / (1+ 2*1e-5)

        pred = self.predictor(torch.tanh(logitC))
        return pred, logitC, pC
    

class SL_R(torch.nn.Module):
    def __init__(self, hidden_neurons, args):
        super(SL_R, self).__init__()
        self.hidden = hidden_neurons
        self.encoder = SL(hidden_neurons, args)
        self.decoder = MLPRecon(hidden_neurons)
        self.device = 'cpu'
        self.temp = 2
    def forward(self, x):
        pred, logitC, pC = self.encoder(x)
            
        C = torch.zeros((8,6), device=self.device)
        for i in range(3):
            C[:, 2*i]   = logitC[:,i] / 2
            C[:, 2*i+1] = -logitC[:,i] / 2
        C[:,:2]  = F.gumbel_softmax(C[:,:2],  tau=1, hard=True, dim=1)
        C[:,2:4] = F.gumbel_softmax(C[:,2:4], tau=1, hard=True, dim=1)
        C[:,4:]  = F.gumbel_softmax(C[:,4:],  tau=1, hard=True, dim=1)

        recon = self.decoder(C[:, [1,3,5]])

        return pred, recon, logitC, pC

    def update_temp(self):
        self.temp = self.temp*0.95