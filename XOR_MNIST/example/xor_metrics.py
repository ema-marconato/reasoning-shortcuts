import numpy as np
import torch

from sklearn.metrics import accuracy_score

def extract_statistics(G, Y, C, pred):
    gt = np.zeros(8)
    cpred = np.zeros(8)
    for i in range(3):
        k = 2*(2-i) if i !=2 else 1
        gt = gt + (k) * G[:,i]
        cpred = cpred + (k) * np.around(C[:, i])

    ypred = np.around(pred)

    return accuracy_score(Y, ypred), accuracy_score(gt, cpred), [gt,Y, cpred, ypred]