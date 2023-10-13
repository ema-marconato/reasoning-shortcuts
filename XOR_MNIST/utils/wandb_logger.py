import wandb
import numpy as np
import torch


def wandb_log_step(i, epoch, loss, losses=None):
    wandb.log({"loss":loss, "epoch": epoch,"step": i})
    if losses is not None:
        wandb.log(losses)

def wandb_log_epoch(**kwargs):
    # log accuracies
    epoch = kwargs['epoch']
    acc = kwargs['acc']
    c_acc = kwargs['cacc']
    wandb.log({'acc': acc, "c-acc":c_acc, 'epoch':epoch})

    lr = kwargs['lr']
    wandb.log({'lr':lr})

    tloss = kwargs['tloss']
    wandb.log({'test-loss':tloss})
    
    # # log specific to each concept
    # complete_distr = kwargs['complete_distr']

    # L = complete_distr.shape[1]

    # for i in range(L):
    #     wandb.log({'acc_c=%i'%i: complete_distr[t,i], "task":t})
    #     # wandb.log({'align_%i'%i: dis_h[i], 'task': t})
    #     # wandb.log({'log_lh_%i'%i:  log_lh_c[i], ' task': t})

    # # log confusion matrices
    # cf_labels, cf_preds, cf_concepts, cf_z_pred = kwargs['cf']
    # K = cf_labels.max()
    # wandb.log({'confusion-preds': wandb.plot.confusion_matrix(None, cf_labels, cf_preds,  class_names=[str(i) for i in range(K+1)]),
    #             'task': t})

    # if len(cf_concepts.shape) == 1:
    #     M = cf_concepts.max()
    #     wandb.log({'confusion-concepts': wandb.plot.confusion_matrix(None, cf_concepts, cf_z_pred, class_names=[str(i) for i in range(M+1)])})
    # else:
    #     mask = (cf_concepts[:,0] == 2*t) | (cf_concepts[:,0] == 2*t + 1)
    #     mask = mask & ((cf_concepts[:,1] == 2*t) | (cf_concepts[:,1] == 2*t + 1))

    # wandb.log({'confusion-preds': wandb.plot.confusion_matrix(None, cf_labels, cf_preds,  class_names=[str(i) for i in range(K+1)]),
    #             'confusion-concepts': wandb.plot.confusion_matrix(None, cf_concepts, cf_z_pred, class_names=[str(i) for i in range(M+1)]),
    #             'task': t})

    # # log perfs on test-set
    # wandb.log({"overall-acc-labels": kwargs['acc_labels']})
    # wandb.log({"overall-acc-concepts": kwargs['acc_concepts']})


def wand_log_end(t_acc, t_c_acc):
    # log score metrics
    wandb.log({'test-acc': t_acc, 'test-c_acc': t_c_acc})
