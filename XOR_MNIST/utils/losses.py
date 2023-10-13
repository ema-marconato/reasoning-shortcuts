import torch
import numpy as np
import torch.nn.functional as F

def ADDMNIST_Classification(out_dict: dict, args):
    out = out_dict['YS']
    labels = out_dict['LABELS'].to(torch.long)

    if args.model in ['mnistdpl', 'mnistdplrec']:
        loss = F.nll_loss(out.log(), labels, reduction='mean')
    elif args.model in ['mnistsl', 'mnistslrec']:
        loss = F.cross_entropy(out, labels, reduction='mean')
    else:
        loss = torch.tensor(1e-5)
        
    assert loss > 0, loss
    
    losses = {'y-loss': loss.item()}

    return loss, losses


def ADDMNIST_Concept_Match(out_dict: dict, args):
    reprs = out_dict['CS']
    concepts = out_dict['CONCEPTS'].to(torch.long)
    objs = torch.split(reprs, 1, dim=1)
    g_objs = torch.split(concepts, 1, dim=1)
    loss = torch.tensor(0.,device=reprs.device)

    assert len(objs) == len(g_objs), f'{len(objs)}-{len(g_objs)}'

    for j in range(len(objs)):
        mask = (g_objs[j] != -1)
        if mask.sum() > 0:
            loss += torch.nn.CrossEntropyLoss()(objs[j][mask].squeeze(1), g_objs[j][mask].view(-1))
    losses = {'c-loss': loss.item()}

    return loss / len(objs), losses

def ADDMNIST_REC_Match(out_dict: dict, args):
    recs, inputs, mus, logvars  = out_dict['RECS'], out_dict['INPUTS'], out_dict['MUS'], out_dict['LOGVARS']
    
    L = len(recs)
    
    assert inputs.size() == recs.size(), f'{len(inputs)}-{len(recs)}'

    recon = F.binary_cross_entropy(recs.view(L, -1), inputs.view(L,-1))
    kld   =  (-0.5 * (1 + logvars - mus ** 2 - logvars.exp()).sum(1).mean() - 1).abs()

    losses = {'recon-loss': recon.item(), 'kld':kld.item()}
    
    return recon + args.beta * kld, losses 

def ADDMNIST_Entropy(out_dict, args):
    pCs = out_dict['pCS']
    l = pCs.size(-1)
    p_mean = torch.mean(pCs, dim=0).view(-1,l)
    
    ## ADD SMALL OFFSET
    p_mean += 1e-5
    
    with torch.no_grad():
        Z = torch.sum(p_mean, dim=1, keepdim=True)
    p_mean /= Z
            
    loss = 0 
    for i in range(p_mean.size(0)):
        loss -= torch.sum( p_mean[i] * p_mean[i].log() ) / np.log(10) / p_mean.size(0)
    
    losses = {'H-loss': 1-loss }
        
    assert (1-loss) > 0, loss
    
    return 1- loss, losses

def ADDMNIST_rec_class(out_dict:dict, args):
    loss1, losses1 = ADDMNIST_Classification(out_dict, args)
    loss2, losses2 = ADDMNIST_REC_Match(out_dict, args)

    losses1.update( losses2)
    
    return loss1+ args.gamma*loss2, losses1

def ADDMNIST_Cumulative(out_dict:dict, args):
    loss, losses   = ADDMNIST_Classification(out_dict,args)
    mitigation = 0
    if args.model in ['mnistdplrec', 'mnistslrec', 'mnistltnrec']:
        loss1, losses1 = ADDMNIST_REC_Match(out_dict, args)
        mitigation += args.w_rec * loss1
        losses.update(losses1) 
    if args.entropy:
        loss2, losses2 = ADDMNIST_Entropy(out_dict, args)
        mitigation += args.w_h * loss2
        losses.update(losses2) 
    if args.c_sup > 0:
        loss3, losses3 = ADDMNIST_Concept_Match(out_dict, args)
        mitigation += args.w_c * loss3
        losses.update(losses3)
        
    return loss + args.gamma * mitigation, losses