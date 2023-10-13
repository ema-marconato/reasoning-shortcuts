import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def evaluate_mix(true, pred):
    ac = accuracy_score(true, pred)
    f1 = f1_score(true, pred, average='weighted')
    # pc = precision_score(true, pred)
    # rc = recall_score(true, pred)
    
    return ac, f1 #, pc, rc 

def evaluate_metrics(model, loader, args, last=False):
    L = len(loader)
    tloss, cacc, yacc = 0, 0, 0
    for i, data in enumerate(loader):
        images, labels, concepts = data
        images, labels, concepts = images.to(model.device), labels.to(model.device), concepts.to(model.device)

        out_dict = model(images)
        out_dict.update({'INPUTS': images, 'LABELS': labels, 'CONCEPTS': concepts})
        
        if last and i == 0:
            y_true = labels.detach().cpu().numpy()
            c_true = concepts.detach().cpu().numpy()
            y_pred = out_dict['YS'].detach().cpu().numpy()
            c_pred = out_dict['CS'].detach().cpu().numpy()
        elif last and i > 0:            
            y_true = np.concatenate([y_true, labels.detach().cpu().numpy()], axis=0)
            c_true = np.concatenate([c_true, concepts.detach().cpu().numpy()], axis=0)
            y_pred = np.concatenate([y_pred, out_dict['YS'].detach().cpu().numpy()], axis=0)
            c_pred = np.concatenate([c_pred, out_dict['CS'].detach().cpu().numpy()], axis=0)  
                
        if args.dataset in ['addmnist', 'shortmnist', 'restrictedmnist'] and not last:
            loss, ac, acc = ADDMNIST_eval_tloss_cacc_acc(out_dict, concepts)
        else:
            NotImplementedError()
        if not last:
            tloss += loss.item()
            cacc  += ac
            yacc  += acc
    if last:            
        ys = np.argmax(y_pred, axis=1)
        gs = np.split(c_true, c_true.shape[1], axis=1)
        cs = np.split(c_pred, c_pred.shape[1], axis=1)
                
        assert len(gs) == len(cs), f'gs: {gs.shape}, cs: {cs.shape}'
                
        gs = np.concatenate(gs, axis=0).squeeze(1)
        cs = np.concatenate(cs, axis=0).squeeze(1).argmax(axis=1)
        
        assert gs.shape == cs.shape, f'gs: {gs.shape}, cs: {cs.shape}'
                
        return y_true, gs, ys, cs
    else:          
        return tloss / L, cacc / L, yacc /L

def ADDMNIST_eval_tloss_cacc_acc(out_dict, concepts):
    reprs = out_dict['CS']
    L = len(reprs)
    
    objs = torch.split(reprs, 1, dim=1, )
    g_objs = torch.split(concepts, 1, dim=1)
    
    assert len(objs) == len(g_objs), f'{len(objs)}-{len(g_objs)}'
        
    loss, cacc = 0, 0
    for j in range(len(objs)):
        # enconding + ground truth
        obj_enc = objs[j].squeeze(dim=1)
        g_obj   = g_objs[j].to(torch.long).view(-1)
         
        # evaluate loss 
        loss += torch.nn.CrossEntropyLoss()(obj_enc, g_obj)
        
        # concept accuracy of object
        c_pred = torch.argmax(obj_enc, dim=1)
        
        assert c_pred.size() == g_obj.size(), f'size c_pred: {c_pred.size()}, size g_objs: {g_obj.size()}'
        
        correct = (c_pred == g_obj).sum().item()
        cacc += correct / len(objs[j])
        
    y = out_dict['YS']
    y_true = out_dict['LABELS']

    y_pred = torch.argmax(y, dim=-1)
    assert y_pred.size() == y_true.size(), f'size c_pred: {c_pred.size()}, size g_objs: {g_obj.size()}' 

    acc = (y_pred == y_true).sum().item() / len(y_true)
    
    return loss / len(objs), cacc / len(objs) * 100, acc * 100
