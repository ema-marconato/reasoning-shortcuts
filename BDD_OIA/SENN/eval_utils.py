# -*- coding: utf-8 -*-
""" Code for training and evaluating Self-Explaining Neural Networks.
Copyright (C) 2018 David Alvarez-Melis <dalvmel@mit.edu>

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License,
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.
"""
import sys
import pdb
import numpy as np
from tqdm import tqdm
import torch
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#===============================================================================
#=============       CONTINUOUS SPACE VERSIONS       ===========================
#===============================================================================

def local_lipschitz_estimate(f,x,mode=1,eps = None, tol = 1e-12, maxit = 1e6,
                                patience = 3, log_interval = 10, verbose = False):
    """
        Compute one-sided lipschitz estimate for GSENN model. Adequate for local
         Lipschitz, for global must have the two sided version. This computes:

            max_z || f(x) - f(z)|| / || x - z||

        Instead of:

            max_z1,z2 || f(z1) - f(z2)|| / || z1 - z2||

        If eps provided, does local lipzshitz in ball of radius eps.

        Mode 1: max_z || f(x) - f(z)|| / || x - z||   , with f = theta
        Mode 2: max_z || f(x) - f(z)|| / || g(x) - g(z)||, with f = theta, g = h

    """
    norm_lambda = 1e1
    f.eval()

    cuda = x.is_cuda

    x = Variable(x.data, requires_grad = False)
    if cuda:
        x = x.cuda()
    if eps is not None:
        # Start close to x!
        noise_vec = eps*torch.randn(x.size())
        if cuda:
            noise_vec = noise_vec.cuda()
        z = Variable(x.data.clone() + noise_vec, requires_grad = True)
        if mode == 1:
            progress_string = "\rStep: {:8}/{:8} Loss:{:8.2f} L:{:5.2f} ||x-z||:{:8.2f} Improv.:{:6.2f}"
        else:
            progress_string = "\rStep: {:8}/{:8} Loss:{:8.2f} L:{:5.2f} ||gx-gz||:{:8.2f} Improv.:{:6.2f}"
    else:
        z = Variable(torch.randn(x.size()), requires_grad = True)
        if cuda:
            z = z.cuda()
        progress_string = "\rStep: {:8}/{:8} L:{:5.2f} Improv.:{:6.2f}"


    if mode == 1:
        # fx = f(x).detach()
        # fz = f(z)
        print("test")
        _ = f(x)
        fx = f.thetas.detach()
        _ = f(z)
        fz = f.thetas
    else:
        _ = f(x)
        fx = f.thetas.detach()
        gx = f.concepts.detach()
        _ = f(z)
        fz = f.thetas
        gz = f.concepts

    optim = torch.optim.SGD([z], lr=0.01)

    print("end of today")
    sys.exit(1)
    i = 0
    improvements = [tol*2]
    prev_lip = 0 #(((y_hat - y).norm())/((x-z).norm())).data[0]
    prev_loss = 0 #

    while True:
        i += 1
        optim.zero_grad()
        if mode == 1:
            _ = f(z)
            fz = f.thetas
            dist_f = (fz - fx).norm()
            dist_x = ( z -  x).norm()
            loss   = dist_x/dist_f  # Want to maximize d_f/d_x (reciprocal)
        else:
            _ = f(z)
            fz = f.thetas
            gz = f.concepts
            dist_f = (fz - fx).norm()
            dist_g = (gz - gx).norm()
            loss   = dist_g/dist_f  # Want to maximize d_f/d_g

        lip = 1/loss.data[0]

        # Introduce ball constraint with lagrangean
        if eps is not None:
            #ball_loss = F.relu(dist_g - eps)
            if mode == 1:
                dist = dist_x.data[0]
                loss += norm_lambda*F.relu(dist_x - eps)
            else:
                dist = dist_g.data[0]
                loss += norm_lambda*F.relu(dist_g - eps)

        loss.backward()
        optim.step()

        # a last correction...
        #input_param.data.clamp_(0, 1)

        improvements.append(prev_loss - loss.data[0])
        prev_loss = loss.data[0]
        #print()
        #print(improvements[-1])

        if i % log_interval == 0:
            if eps is not None:
                prog_list = [i, maxit, loss.data[0], lip, dist, improvements[-1]]
            else:
                prog_list = [i, maxit, lip, improvements[-1]]

            print(progress_string.format(*prog_list))#, end = '')

        if (i > 10) and (max(improvements[-patience:]) < - tol):
            # Best improvement is negative and below tol threshold, i.e. all prev k steps wrosening > tol
            #print()
            #print(improvements[-patience:])
            if verbose: print('\nReached stop condition: improvement stalled for {} iters.'.format(patience))
            break
        if (i > maxit):
            if verbose: print('\nReached stop condition: maximum number of iterations ({}).'.format(maxit))
            break

    print()
    print('Estimated Lipschitz constant: {:8.2f}'.format(lip))
    if eps is not None and verbose:
        if mode == 1:
            print('|| x - z || = {:8.2f} < {:8.2f}'.format((z-x).norm().data[0], eps))
        else:
            print('|| g(x) - g(z) || = {:8.2f} < {:8.2f}'.format((gz-gx).norm().data[0], eps))

    return lip, z.data

def estimate_dataset_lipschitz(model, dataloader, continuous=True, mode = 1, eps = 1, tol = 1e-2, maxpoints = None,
                              maxit = 1e5, patience = 3, log_interval = 10, cuda= False, verbose = False):
    """
        Continuous and discrete space version.

    """
    model.eval()
    Lips = []
    # ToDoL Add a parfor here
    

    for i, (inputs, targets) in enumerate(dataloader, 0):
        if cuda:
            inputs = inputs.cuda()
        #print(inputs.size())
        #print(asd.asd)
        inputs = Variable(inputs) #targets = Variable(inputs), Variable(targets)
        l,_ = local_lipschitz_estimate(model, inputs, mode = mode, eps=eps, tol=tol,
                                      maxit=maxit, log_interval=log_interval,
                                      patience = patience, verbose = verbose)
        Lips.append(l)
        #print('Warning: early break')
        #break
        if maxpoints is not None and i == maxpoints:
            break
    Lips = np.array(Lips)
    return Lips.mean(), Lips


#===============================================================================
#=============       EMPIRICAL SAMPLE VERSIONS       ===========================
#===============================================================================

def pairwise_distances(x, y=None):
    '''
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''
    x_norm = (x**2).sum(1).view(-1, 1)
    if y is not None:
        y_t = torch.transpose(y, 0, 1)
        y_norm = (y**2).sum(1).view(1, -1)
    else:
        y_t = torch.transpose(x, 0, 1)
        y_norm = x_norm.view(1, -1)

    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    # Ensure diagonal is zero if x=y
    # if y is None:
    #     dist = dist - torch.diag(dist.diag)
    return torch.clamp(dist, 0.0, np.inf)


def sample_local_lipschitz(model, dataset, mode = 2, max_distance = None, top_k = 1, cuda = False):
    """

        For every point in dataset, find pair point y in dataset that maximizes relative variation of model

            MODE 1:     || th(x) - th(y) ||/||x - y||
            MODE 2:     || th(x) - th(y) ||/||h(x) - h(y)||

            - dataset: a tds obkect
            - top_k : how many to return
            - max_distance: maximum distance between points to consider (radius)

        TODO: Takes matrix of distances to avoid recomputation in every step.
        NO POINT, WE NEED H DISTANCES, not x

    """
    model.eval()

    tol = 1e-10 # To avoid numerical problems

    # Create dataloader from tds without shuffle
    dataloader = DataLoader(dataset, batch_size = 128, shuffle=False)
    n = len(dataset) # len(dataset)

    Hs = []
    Ts = []


    for i, (inputs, targets) in enumerate(dataloader):
        # get the inputs
        if cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        input_var = torch.autograd.Variable(inputs, volatile=True)
        target_var = torch.autograd.Variable(targets, volatile=True)

        _ = model(input_var)
        Ts.append(model.thetas.squeeze())
        Hs.append(model.concepts.squeeze())

    Ts = torch.cat(Ts, dim = 0)
    num_dists = pairwise_distances(Ts) # Numerator

    if mode == 1:
        denom_dists = pairwise_distances(dataset)
    if mode == 2:
        Hs = torch.cat(Hs)
        denom_dists = pairwise_distances(Hs)

    ratios = torch.Tensor(n,n)

    if max_distance is not None:
        # Distances above threshold: make them inf
        #print((denom_dists > max_distance).size())
        nonzero = torch.nonzero((denom_dists > max_distance).data).size(0)
        total =  denom_dists.size(0)**2
        print('Number of zero denom distances: {} ({:4.2f}%)'.format(
                total - nonzero, 100*(total-nonzero)/total))
        denom_dists[denom_dists > max_distance] = -1.0 #float('inf')
    # Same with self dists
    denom_dists[denom_dists == 0] = -1.0 #float('inf')
    ratios = (num_dists/denom_dists).data
    argmaxes = {k: [] for k in range(n)}
    vals, inds = ratios.topk(top_k, 1, True, True)
    argmaxes = {i:  [(j,v) for (j,v) in zip(inds[i,:],vals[i,:])] for i in range(n)}
    return vals[:,0].numpy(), argmaxes

    #
    #
    # n = len(dataset) # len(dataset)
    # ratios = {}
    # for i in tqdm(range(n)):
    #     x = Variable(dataset.data_tensor[i,:]).view(1,-1)
    #     _ = model(x)
    #     Th_x = model.thetas
    #     for j in range(n):
    #         if i == j: continue
    #         y     = Variable(dataset.data_tensor[j,:]).view(1,-1)
    #         ratio, num, denom = lipschitz_ratio(model, x, y, Th_x = Th_x)
    #         if max_distance is not None and denom > max_distance:
    #             continue
    #         ratios[(i,j)] = ratio.data.numpy()
    # out = []
    # for i, (pair, val) in enumerate(sorted(ratios.items(), key=lambda x: x[1], reverse = True)):
    #     out.append((pair, val))
    #     if i + 1 == top_k:
    #         break


def lipschitz_ratio(model, x, y, Th_x = None, mode = 1):
    """
            For two points x,z compute:

            MODE 1:     || th(x) - th(y) ||/||x - y||
            MODE 2:     || th(x) - th(y) ||/||h(x) - h(y)||

            If Th_x provided, won't recompute.
    """
    cuda = x.is_cuda
    if Th_x is None:
        x = Variable(x.data, requires_grad = False)
        if cuda:
            x = x.cuda()
        _ = model(x)
        Th_x = model.thetas

    _ = model(y)
    Th_y = model.thetas
    num = (Th_y - Th_x).norm()

    if mode == 1:
        denom = ( y -  x).norm()
    else:
        h_x  = model.concepts
        h_y  = model.concepts
        denom  = (h_x - h_y).norm()

    ratio   =  num/denom
    return ratio,num,denom

def find_maximum_lipschitz_dataset(model, dataset, top_k = 1, max_distance = None):
    """
        Find pair of points x and y in dataset that maximize relative variation of model

        || f(x) - f(x) ||/||x - y||

    """
    model.eval()
    n = len(dataset) # len(dataset)
    ratios = {}
    for i in range(n):
        x = Variable(dataset.data_tensor[i,:]).view(1,-1)
        fx = model(x)
        for j in range(i+1, n):
            y = Variable(dataset.data_tensor[j,:]).view(1,-1)
            fy = model(y)
            dxy = (x-y).norm().data.numpy()
            if max_distance is not None and dxy > max_distance:
                continue
            ratios[(i,j)] = (fx - fy).norm().data.numpy()/dxy
    out = []
    for i, (pair, val) in enumerate(sorted(ratios.items(), key=lambda x: x[1], reverse = True)):
        out.append((pair, val))
        if i + 1 == top_k:
            break
    return out
