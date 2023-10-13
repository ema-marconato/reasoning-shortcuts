import torch
import numpy as np
import itertools
import ltn
from example.ltn_models import LTN

class LogitsToPredicate(torch.nn.Module):
    """
    This model has inside a logits model, that is a model which compute logits for the classes given an input example x.
    The idea of this model is to keep logits and probabilities separated. The logits model returns the logits for an example,
    while this model returns the probabilities given the logits model.

    In particular, it takes as input an example x and a class label d. It applies the logits model to x to get the logits.
    Then, it applies a softmax function to get the probabilities per classes. Finally, it returns only the probability related
    to the given class d.
    """
    def __init__(self, logits_model):
        super(LogitsToPredicate, self).__init__()
        self.logits_model = logits_model
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x, d):
        logits = self.logits_model(x)
        probs = self.softmax(logits)
        out = torch.gather(probs, 1, d)
        return out

def shannon_entropy(pC):
    H = 0

    for i in range(3):
        aC = torch.mean(pC, dim=0)
        H += - ( aC[i] * torch.log(aC[i]) + (1-aC[i]) * torch.log( 1 - aC[i])  ) / 3

    return H / np.log(2)

def semantic_loss(pC, Y):
    if len(Y.shape) == 2:
        pY = torch.softmax(Y, dim=1)
    else:
        pY= torch.zeros((8,2))
        pY[:,0] = 1 - Y 
        pY[:,1] = Y 

    loss = 0 
    for i,j,k in itertools.product(range(2), range(2), range(2)):
        if (i + j + k) %2 == 0: 
            loss += pY[:,0] * pC[:, 0+i] * pC[:,2+j] * pC[:, 4+k]
        else: 
            loss += pY[:,1] * pC[:, 0+i] * pC[:,2+j] * pC[:, 4+k]
    loss += 1e-5
    return - loss.log().mean()

def sat_agg_loss(model:LTN, p1, p2, p3, labels, grade):

    # convert to variables 
    bit1   = ltn.Variable('bit1', p1)#, trainable=True)
    bit2   = ltn.Variable('bit2', p2)#, trainable=True)
    bit3   = ltn.Variable('bit3', p3)#, trainable=True)
    y_true = ltn.Variable('labels',labels)
    
    # variables in LTN
    b_1 = ltn.Variable("b_1", torch.tensor(range(2)))
    b_2 = ltn.Variable("b_2", torch.tensor(range(2)))
    b_3 = ltn.Variable("b_3", torch.tensor(range(2)))

    And    = ltn.Connective(ltn.fuzzy_ops.AndProd())
    Exists = ltn.Quantifier(ltn.fuzzy_ops.AggregPMean(p=2), quantifier="e")
    Forall = ltn.Quantifier(ltn.fuzzy_ops.AggregPMeanError(p=2), quantifier="f")
    SatAgg = ltn.fuzzy_ops.SatAgg()

    sat_agg = Forall(
                    ltn.diag(bit1, bit2, bit3, y_true),
                    Exists(
                        [b_1, b_2, b_3],
                        And(And(  model.ltn(bit1,b_1), model.ltn(bit2,b_2) ), 
                            model.ltn(bit3, b_3)
                            ),
                        cond_vars=[b_1, b_2, b_3, y_true],
                        cond_fn=lambda b_1, b_2, b_3, z: torch.eq((b_1.value + b_2.value + b_3.value)%2, z.value),
                        p=grade
                    )).value
    return 1 - sat_agg
