import ltn
import torch
import itertools

class identity(torch.nn.Module):
    def __init__(self):
        super(identity,self).__init__()
    def forward(self,x,d):
        return torch.gather(x, 1, d)

class ADDMNIST_SAT_AGG(torch.nn.Module):
    def __init__(self, loss, task='addition', nr_classes=19) -> None:
        super().__init__()
        self.base_loss = loss
        self.task = task
        if task == 'addition':
            self.nr_classes = 19
        elif task == 'product':
            self.nr_classes = 81
        elif task == 'multiop':
            self.nr_classes = 15

        self.iden = identity()
        self.ltn = ltn.Predicate(self.iden)

        self.grade = 1
    
    def update_grade(self, epoch):
        if (epoch) % 3 == 0 and epoch != 0:
            self.grade += 2 if self.grade < 10 else 0


    def forward(self, out_dict, args):
        
        loss, losses = self.base_loss(out_dict, args)

        # load from dict
        Ys = out_dict['LABELS']
        pCs = out_dict['pCS']

        prob_digit1, prob_digit2 = pCs[:, 0, :], pCs[:, 1,:]

        if self.task == 'addition':
            sat_loss = ADDMNISTsat_agg_loss(self.ltn, prob_digit1, prob_digit2, Ys, self.grade )
        elif self.task == 'product':
            sat_loss = PRODMNISTsat_agg_loss(self.ltn, prob_digit1, prob_digit2, Ys, self.grade )
        elif self.task == 'multiop':
            sat_loss = MULTIOPsat_agg_loss(self.ltn, prob_digit1, prob_digit2, Ys, self.grade )
        
        losses.update({'sat-loss':sat_loss.item()})

        return loss + sat_loss, losses

def ADDMNISTsat_agg_loss(eltn, p1, p2, labels, grade):

    max_c = p1.size(-1)

    # convert to variables 
    bit1   = ltn.Variable('bit1', p1)#, trainable=True)
    bit2   = ltn.Variable('bit2', p2)#, trainable=True)
    y_true = ltn.Variable('labels',labels)
    
    # variables in LTN
    b_1 = ltn.Variable("b_1", torch.tensor(range(max_c)))
    b_2 = ltn.Variable("b_2", torch.tensor(range(max_c)))

    And    = ltn.Connective(ltn.fuzzy_ops.AndProd())
    Exists = ltn.Quantifier(ltn.fuzzy_ops.AggregPMean(p=2), quantifier="e")
    Forall = ltn.Quantifier(ltn.fuzzy_ops.AggregPMeanError(p=2), quantifier="f")
    SatAgg = ltn.fuzzy_ops.SatAgg()

    # print(torch.eq((b_1.value + b_2.value), y_true.value).size())
    # quit()
    sat_agg = Forall(
                    ltn.diag(bit1, bit2,y_true),
                    Exists(
                        [b_1, b_2],
                        And(eltn(bit1,b_1), eltn(bit2,b_2)),
                        cond_vars=[b_1, b_2, y_true],
                        cond_fn=lambda d1, d2, z: torch.eq((d1.value + d2.value), z.value),
                        p=grade
                    )).value
    return 1 - sat_agg

def PRODMNISTsat_agg_loss(eltn, p1, p2, labels, grade):
    max_c = p1.size(-1)

    # convert to variables 
    bit1   = ltn.Variable('bit1', p1)#, trainable=True)
    bit2   = ltn.Variable('bit2', p2)#, trainable=True)
    y_true = ltn.Variable('labels',labels)

    # print(bit1)
    # print(bit2)
    # print(y_true)
    
    # variables in LTN
    b_1 = ltn.Variable("b_1", torch.tensor(range(max_c)))
    b_2 = ltn.Variable("b_2", torch.tensor(range(max_c)))


    And    = ltn.Connective(ltn.fuzzy_ops.AndProd())
    Exists = ltn.Quantifier(ltn.fuzzy_ops.AggregPMean(p=2), quantifier="e")
    Forall = ltn.Quantifier(ltn.fuzzy_ops.AggregPMeanError(p=2), quantifier="f")
    SatAgg = ltn.fuzzy_ops.SatAgg()

    sat_agg = Forall(
                    ltn.diag(bit1, bit2,y_true),
                    Exists(
                        [b_1, b_2],
                        And(eltn(bit1,b_1), eltn(bit2,b_2)),
                        cond_vars=[b_1, b_2, y_true],
                        cond_fn=lambda b_1, b_2, z: torch.eq( b_1.value * b_2.value, z.value),
                        p=grade
                    )).value
    return 1 - sat_agg

def MULTIOPsat_agg_loss(eltn, p1, p2, labels, grade):

    max_c = p1.size(-1)

    # convert to variables 
    bit1   = ltn.Variable('bit1', p1)#, trainable=True)
    bit2   = ltn.Variable('bit2', p2)#, trainable=True)
    y_true = ltn.Variable('labels',labels)

    # variables in LTN
    b_1 = ltn.Variable("b_1", torch.tensor(range(max_c)))
    b_2 = ltn.Variable("b_2", torch.tensor(range(max_c)))


    And    = ltn.Connective(ltn.fuzzy_ops.AndProd())
    Exists = ltn.Quantifier(ltn.fuzzy_ops.AggregPMean(p=2), quantifier="e")
    Forall = ltn.Quantifier(ltn.fuzzy_ops.AggregPMeanError(p=2), quantifier="f")
    SatAgg = ltn.fuzzy_ops.SatAgg()

    sat_agg = Forall(
                    ltn.diag(bit1, bit2,y_true),
                    Exists(
                        [b_1, b_2],
                        And(eltn(bit1,b_1), eltn(bit2,b_2)),
                        cond_vars=[b_1, b_2, y_true],
                        cond_fn=lambda b_1, b_2, z: torch.eq(b_1.value**2 + b_2.value**2 + b_1.value * b_2.value, 
                                                             z.value),
                        p=grade
                    )).value
    return 1 - sat_agg