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

import os
import pdb
import numpy as np

import matplotlib.pyplot as plt
import pprint # For feature explainer

import torch
from torch.autograd import Variable
from torchvision.utils import make_grid
import matplotlib.gridspec as gridspec

pp = pprint.PrettyPrinter(indent=4)


def generate_dir_names(dataset, args, make = True):
    
    C = '_csup'    if args.h_labeled_param > 0 else ''
    H = '_entropy' if args.w_entropy > 0       else ''
    suffix = f'{args.model_name+C+H}-{args.seed}'

    # if args.h_type == 'input':
    #     suffix = '{}_H{}_Th{}_Reg{:0.0e}_LR{}'.format(
    #                 args.theta_reg_type,
    #                 args.h_type,
    #                 args.theta_arch,
    #                 args.theta_reg_lambda,
    #                 args.lr,
    #                 )
    # else:
    #     suffix = '{}_H{}_Th{}_Cpts{}_Reg{:0.0e}_Sp{}_LR{}_hlp{}'.format(
    #                 args.theta_reg_type,
    #                 args.h_type,
    #                 args.theta_arch,
    #                 args.nconcepts,
    #                 args.theta_reg_lambda,
    #                 args.h_sparsity,
    #                 args.lr,
    #                 args.h_labeled_param,
    #                 )

    model_path     = os.path.join(args.model_path, dataset, suffix)
    log_path       = os.path.join(args.log_path, dataset, suffix)
    results_path   = os.path.join(args.results_path, dataset, suffix)

    if make:
        for p in [model_path, results_path]: #, log_path,
            if not os.path.exists(p):
                os.makedirs(p)

    return model_path, log_path, results_path


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



### Animation Utils

# animation function
def animate_fn(i, xx,yy,Ts, Cs):
    t = Ts[i]
    C = Cs[i]
    cont = plt.contourf(xx, yy, C, 25, cmap = plt.cm.RdBu)
    plt.title(r'Iter = %i' % t)
    return cont

def animate_training(Steps, Cs, X_train, y_train):
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    h = .02  # step size in the mesh
    x_min, x_max = X_train[:, 0].min() - .5, X_train[:, 0].max() + .5
    y_min, y_max = X_train[:, 1].min() - .5, X_train[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    line, = ax.plot([],[], '-')
    line2, = ax.plot([],[],'--')
    ax.set_xlim(np.min(xx), np.max(xx))
    ax.set_xlim(np.min(yy), np.max(yy))
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
       edgecolors='k')
    #ax.contourf(xx, yy, Cs[0].reshape(xx.shape))

    anim = animation.FuncAnimation(fig, animate_fn, frames = len(Steps), fargs = (xx,yy,Steps,Cs,), interval = 200, blit = False)

    return anim

# Got these two from scikit learn embedding example
def make_meshgrid(x, y, h=.02):
    """Create a mesh of points to plot in

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(model, X, **params):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    fig, ax = plt.subplots()
    Z = model(X)
    Z = Z.data.numpy()
    C = np.argmax(Z,axis=1)
    C = C.reshape(xx.shape)
    out = ax.contourf(xx, yy, C, **params)
    return out


def plot_embedding(X,y,Xp, title=None):
    """ Scale and visualize the embedding vectors """
    x_min, x_max = np.min(Xp, 0), np.max(Xp, 0)
    Xp = (Xp - x_min) / (x_max - x_min)

    plt.figure(figsize=(20,10))
    ax = plt.subplot(111)
    for i in range(Xp.shape[0]):
        plt.text(Xp[i, 0], Xp[i, 1], str(y[i]),
                 color=plt.cm.Set1(y[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})

    if hasattr(offsetbox, 'AnnotationBbox'):
        # only print thumbnails with matplotlib > 1.0
        shown_images = np.array([[1., 1.]])  # just something big
        for i in range(X.shape[0]):
            dist = np.sum((Xp[i] - shown_images) ** 2, 1)
            if np.min(dist) < 4e-3:
                # don't show points that are too close
                continue
            shown_images = np.r_[shown_images, [Xp[i]]]
            imagebox = offsetbox.AnnotationBbox(
                offsetbox.OffsetImage(X[i,:].reshape(28,28), cmap=plt.cm.gray_r),
                Xp[i])
            ax.add_artist(imagebox)
    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)


def _explain_class(model, x_raw, x,k,typ='pos',thresh = 0.5,recompute=True):
    """
        Given an input x and class index k, explain f(x) by returning indices of
        features in x that have highest positive impact on predicting class k.
    """
    if recompute:
        y = model(x) # y = self.model(x)
    B_k = model.params[0,k,:].data.numpy()
    if typ == 'pos':
        Mask = (B_k > thresh).astype(np.int).reshape(x.size()).squeeze()
    elif typ == 'neg':
        Mask = (B_k < -thresh).astype(np.int).reshape(x.size()).squeeze()
    else:
        # Return weights instead of mask
        return B_k.reshape(x.size()).squeeze()
    Masked_x = Mask*x_raw.numpy().squeeze()
    return Masked_x

def explain_digit(model, x_raw, x, thresh = 0.5, save_path = None):
    """
        Given an input x, explain f(x) by returning indices of
        features in x that have highest positive impact on predicting each class.

        x_raw is passed for plotting purposes
    """
    plt.imshow(x_raw.squeeze().numpy())
    plt.title('Input:')
    plt.xticks([])
    plt.yticks([])
    if save_path:
        plt.savefig(save_path+'_input.pdf',  bbox_inches = 'tight', format='pdf', dpi=300)
    plt.show()
    y_pred = model(x)

    pred_class = np.argmax(y_pred.data.numpy())
    print('Predicted: ',pred_class)

    fig, ax = plt.subplots(3,model.dout,figsize=(1.5*model.dout,1.5*3))
    for i in range(model.dout):
        #print('Class {}:'.format(i))

        # Positive
        x_imask = _explain_class(model, x_raw, x,i,typ ='pos', recompute=False, thresh = thresh)
        ax[0,i].imshow(x_imask)
        ax[0,i].set_xticks([])
        ax[0,i].set_yticks([])
        ax[0,i].set_title('Class: {}'.format(i))

        # Negative
        x_imask = _explain_class(model, x_raw, x,i,  typ ='neg', recompute=False, thresh = thresh)
        ax[1,i].imshow(x_imask)
        ax[1,i].set_xticks([])
        ax[1,i].set_yticks([])

        # Combined
        x_imask = _explain_class(model, x_raw, x,i,  typ ='both', recompute=False)
        ax[2,i].imshow(x_imask, cmap = plt.cm.RdBu)
        ax[2,i].set_xticks([])
        ax[2,i].set_yticks([])
        #print(np.linalg.norm(x_imask))
        #print(x_imask[:5,:5])

        if i == 0:
            ax[0,0].set_ylabel('Pos. Feats.')
            ax[1,0].set_ylabel('Neg. Feats.')
            ax[2,0].set_ylabel('Combined')


    if save_path:
        plt.savefig(save_path + '_expl.pdf', bbox_inches = 'tight', format='pdf', dpi=300)
    plt.show()

def plot_text_explanation(words, values, n_cols = 6, save_path = None):
    import seaborn as sns
    # Get some pastel shades for the colors
    #colors = plt.cm.BuPu(np.linspace(0, 0.5, len(rows)))
    cmap = sns.cubehelix_palette(light=1, as_cmap=True)

    n_rows = int(min(len(values), len(words)) / n_cols) + 1

    # Plot bars and create text labels for the table
    if type(words) is str:
        words = words.split(' ')

    cellcolours = np.empty((n_rows, n_cols), dtype='object')
    celltext    = np.empty((n_rows, n_cols), dtype='object')

    for r in range(n_rows):
        for c in range(n_cols):
            idx = (r * n_cols + c)
            val =  values[idx] if (idx < len(values)) else 0
            cellcolours[r,c] = cmap(val)
            celltext[r,c] = words[idx] if (idx < len(words)) else ''

    fig, ax = plt.subplots()#figsize=(n_cols, n_rows))

    # Hide axes
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)

    # Add a table at the bottom of the axes
    tab = plt.table(cellText=celltext,
                          cellColours = cellcolours,
                          rowLabels=None,
                          rowColours=None,
                          colLabels=None,
                          cellLoc='center',
                          loc='center')

    for key, cell in tab.get_celld().items():
        cell.set_linewidth(0)

    tab.set_fontsize(14)
    tab.scale(1.5, 1.5)  # may help

    # Adjust layout to make room for the table:
    #plt.subplots_adjust(left=0.2, bottom=0.2)

    #plt.ylabel("Loss in ${0}'s".format(value_increment))
    plt.yticks([])
    plt.xticks([])
    plt.title('')
    plt.axis('off')
    plt.grid('off')
    if save_path:
        plt.savefig(save_path + '_expl.pdf', bbox_inches = 'tight', format='pdf', dpi=300)
    plt.show()


class FeatureInput_Explainer():
    """
        Explainer for classification task models that take vector of features
        as input and and return class probs.

        Arguments:


    """
    def __init__(self, feature_names, binary = False, sort_rows = True, scale_values = True):
        super(FeatureInput_Explainer, self).__init__()
        self.features = feature_names
        self.binary   = binary # Whether it is a binary classif task
        self.sort_rows = sort_rows
        self.scale_values = scale_values

    def explain(self, model, x, thresh = 0.5, save_path = None):
        np.set_printoptions(threshold=15, precision = 2)
        #print('input: {}'.format(x.data.numpy()))
        print('Input:')
        pp.pprint(dict(zip(self.features, *x.data.numpy())))
        print('')

        np.set_printoptions()
        y_pred = model(x)
        pred_class = np.argmax(y_pred.data.numpy())
        print('Predicted: ',pred_class)

        # Get data-dependent params
        B = model.thetas[0,:,:].data.numpy() # class x feats

        Pos_Mask = (B > thresh).astype(np.int)#.reshape(x.size()).squeeze()
        Neg_Mask = (B < thresh).astype(np.int)#.reshape(x.size()).squeeze()

        title = r'Relevance Score $\theta(x)$' + (' (Scaled)' if self.scale_values else '')
        if self.binary:
            d = dict(zip(self.features, B[:,0])) # Change to B[0,:] when B model is truly binary
            A = plot_dependencies(d, title= title,
                                     scale_values = self.scale_values,
                                     sort_rows = self.sort_rows)
        else:
            Pos_Feats = {}
            for k in range(B.shape[0]):
                d = dict(zip(self.features, B[k,:])) # Change to B[0,:] when B model is truly binary
                A = plot_dependencies(d, title= title,
                                         scale_values = self.scale_values,
                                         sort_rows = self.sort_rows)
                Neg_Feats = list(compress(self.features, B[k,:] < -thresh))
                Pos_Feats = list(compress(self.features, B[k,:] > thresh))
                print('Class:{:5} Neg: {}, Pos: {}'.format(k, ','.join(Neg_Feats), ','.join(Pos_Feats)))
        if save_path:
            plt.savefig(save_path, bbox_inches = 'tight', format='pdf', dpi=300)
        plt.show()
        print('-'*60)

    def _explain_class(self, x_raw, x,k,typ='pos',feat_names = None, thresh = 0.5,recompute=True):
        """
            Given an input x and class index k, explain f(x) by returning indices of
            features in x that have highest positive impact on predicting class k.
        """
        if recompute:
            y = model(x) # y = self.model(x)
        B_k = model.params[0,k,:].data.numpy()
        #print((B_k > thresh).astype(np.int))

        if feat_names and typ == 'pos':
            # Return masked features instead of values
            return list(compress(feat_names, B_k > thresh ))
        elif feat_names and typ == 'neg':
            return list(compress(feat_names, B_k < thresh ))
        if typ == 'pos':
            Mask = (B_k > thresh).astype(np.int).reshape(x.size()).squeeze()
        elif typ == 'neg':
            Mask = (B_k < -thresh).astype(np.int).reshape(x.size()).squeeze()
        else:
            # Return weights instead of mask
            return B_k.reshape(x.size()).squeeze()

        Masked_x = Mask*x_raw.numpy().squeeze()
        return Masked_x

def plot_dependencies(dictionary_values,
                      pos_color="#ff4d4d",
                      negative_color="#3DE8F7",
                      reverse_values=False,
                      sort_rows =True,
                      scale_values = True,
                      title="",
                      fig_size=(4, 4),
                      ax = None,
                      x = None,
                      digits = 1, prediction_text = None,
                      show_table = False, ax_table = None):
    """ This function was adapted form the fairml python package

        x needed only if show_table = True

        digits: (int) significant digits in table
    """
    # add check to make sure that dependence features are not zeros
    if np.sum(np.array(dictionary_values.values())) == 0.0:
        print("Feature dependence for all attributes equal zero."
              " There is nothing to plot here. ")
        return None

    column_names = list(dictionary_values.keys())
    coefficient_values = list(dictionary_values.values())

    # get maximum
    maximum_value = np.absolute(np.array(coefficient_values)).max()
    if scale_values:
        coefficient_values = ((np.array(coefficient_values) / maximum_value) * 100)

    if sort_rows:
        index_sorted = np.argsort(np.array(coefficient_values))
    else:
        index_sorted = range(len(coefficient_values))[::-1]

    sorted_column_names = list(np.array(column_names)[index_sorted])
    sorted_column_values = list(np.array(coefficient_values)[index_sorted])
    pos = np.arange(len(sorted_column_values)) + 0.7

    # rearrange this at some other point.
    def assign_colors_to_bars(array_values,
                              pos_influence=pos_color,
                              negative_influence=negative_color,
                              reverse=reverse_values):

        # if you want the colors to be reversed for positive
        # and negative influences.
        if reverse:
            pos_influence, negative_influence = (negative_influence,
                                                 pos_influence)

        # could rewrite this as a lambda function
        # but I understand this better
        def map_x(x):
            if x > 0:
                return pos_influence
            else:
                return negative_influence
        bar_colors = list(map(map_x, array_values))
        return bar_colors

    bar_colors = assign_colors_to_bars(coefficient_values, reverse=True)
    bar_colors = list(np.array(bar_colors)[index_sorted])

    #pdb.set_trace()
    if ax is None and not show_table:
        #pdb.set_trace()
        fig, ax = plt.subplots(figsize=fig_size)
    elif ax is None and show_table:
        fig, axes = plt.subplots(1, 2, figsize=fig_size)
        ax_table, ax = axes

    ax.barh(pos, sorted_column_values, align='center', color=bar_colors)
    ax.set_yticks(pos)
    ax.set_yticklabels(sorted_column_names)
    if scale_values:
        ax.set_xlim(-105, 105)
    else:
        pass
        #ax.set_xlim(-1.05, 1.05)
    if title:
        ax.set_title("{}".format(title))

    if show_table and ax_table:
        cell_text = [[('%1.' + str(digits) + 'f') % v] for v in x]
        if prediction_text is None:
            ax_table.axis('off')
        else:
            print('here')
            ax_table.set_xticklabels([])
            ax_table.set_yticklabels([])
            ax_table.set_yticks([])
            ax_table.set_xticks([])
            for side in ['top', 'right', 'bottom', 'left']:
                ax_table.spines[side].set_visible(False)
            ax_table.set_xlabel(prediction_text)

        ax_table.table(cellText=cell_text,
                                  rowLabels=sorted_column_names[::-1],
                                  rowColours=bar_colors[::-1],
                                  colLabels=None,#['Value'],
                                  colWidths=[1],
                                  loc='left', cellLoc = 'right',
                                  bbox=[0.2, 0.025, 0.95, 0.95])
        ax_table.set_title('Input Value')
        return ax, ax_table

    return ax


def plot_theta_stability(model, input, pert_type = 'gauss', noise_level = 0.5,
                         samples = 5, save_path = None):
    """ Test stability of relevance scores theta for perturbations of an input.

        If model is of type 1 (i.e. theta is of dimension nconcepts x nclass), visualizes
        the perturbations of dependencies with respect to predicted class.

        If model is of type 1/3 (theta is a vector of size nconcepts), then there's only
        one dimension of theta to visualize.

        Args:
            model (GSENN): GSENN model.
            inputs (list of tensors): Inputs over which stability will be tested. First one is "base" input.

        Returns:
            stability: scalar.

        Displays plot also.

    """
    def gauss_perturbation(x, scale = 1):
        noise = Variable(scale*torch.randn(x.size()), volatile = True)
        if x.is_cuda:
            noise = noise.cuda()
        return x + noise

    model.eval()

    # Generate perturbations
    inputs = [input]
    for i in range(samples):
        inputs.append(gauss_perturbation(input, scale=noise_level))

    fig, ax = plt.subplots(2,len(inputs),figsize=(2*len(inputs),1.5*3))

    # Map Them
    thetas = []
    dists  = []
    for i,x in enumerate(inputs):
        pred = model(x)
        ax[0,i].imshow(x.data.cpu().numpy().squeeze())#, cmap = 'Greys', interpolation = 'nearest')
        ax[0,i].set_xticks([])
        ax[0,i].set_yticks([])
        if i == 0:
            ax[0,i].set_title('Original'.format(i))
        else:
            ax[0,i].set_title('Perturbation {}'.format(i))



        theta = model.thetas.data.cpu().numpy().squeeze()
        if theta.shape[1] > 1:
            # Means this is model 1, scalar h and theta_i vector-sized. Choose which one ti visualize
            klass = pred.data.max(1)[1] # Predicted class
            deps = theta[:,klass].squeeze()
            thetas.append(deps)
        else:
            deps = theta
            thetas.append(deps)
        classes = ['C' + str(i) for i in range(theta.shape[0])]
        d = dict(zip(classes, deps))
        A = plot_dependencies(d, title= 'Dependencies', sort_rows = False, ax = ax[1,i])
        #ax[1,i].locator_params(axis = 'y', nbins=10)

        # max_yticks = 10
        # yloc = plt.MaxNLocator(max_yticks)
        # ax[1,i].yaxis.set_major_locator(yloc)
        #print(thetas[-1])
        if i > 0:
            dists.append(np.linalg.norm(thetas[0] - deps))

    dists = np.array(dists)
    plt.tight_layout()
    #print(dists.max())
    if save_path:
        plt.savefig(save_path, bbox_inches = 'tight', format='pdf', dpi=300)
    #plt.show(block=False)


#def concept_grid(model, data_loader, cuda=False, top_k = 6, layout = 'vertical', return_fig=False, save_path = None):
def concept_grid(model, pretrained_model, data_loader, device, top_k = 6, layout = 'vertical', return_fig=False, save_path = None):
    """
        Finds examples in data_loader that are most representatives of concepts.

        For scalar concepts, activation is simply the value of the concept.
        For vector concepts, activation is the norm of the concept.

    """
    print('Warning: make sure data_loader passed to this function doesnt shuffle data!!')
    all_norms = []
    num_concepts = model.parametrizer.nconcept
    concept_dim  = model.parametrizer.dout

    top_activations = {k: np.array(top_k*[-1000.00]) for k in range(num_concepts)}
    top_examples = {k: top_k*[None] for k in range(num_concepts)}
    all_activs = []
    for idx, (data, target, _) in enumerate(data_loader):
        # get the inputs
        #print(data.shape, target.shape, cuda)
        #torch.Size([128, 1, 28, 28]) torch.Size([128])
        """
        if cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        """
        data, target = Variable(data).to(device), Variable(target).to(device)

        #data, target = Variable(data, volatile=True), Variable(target)
        pretrained_out = pretrained_model(data)
        data = pretrained_out

        output = model(data)
        concepts = model.concepts.data
        #pdb.set_trace()
        #concepts[concepts < 0] = 0.0 # This is unncessary if output of H is designed to be > 0.
        if concepts.shape[-1] > 1:
            print('ERROR')
            print(asd.asd)
            activations = np.linalg.norm(concepts, axis = 2)
        else:
            activations = concepts

        all_activs.append(activations)
        # if idx == 10:
        #     break

    all_activs = torch.cat(all_activs)
    top_activations, top_idxs = torch.topk(all_activs, int(top_k/2), 0)
    low_activations, low_idxs = torch.topk(-all_activs, int(top_k/2), 0)
    top_activations = top_activations.squeeze().t()
    low_activations = low_activations.squeeze().t()
    top_idxs = top_idxs.squeeze().t()
    low_idxs = low_idxs.squeeze().t()
    top_examples = {}
    top_idxs = torch.cat([top_idxs,low_idxs],dim=1)
    for i in range(num_concepts):
        buf = data_loader.dataset[top_idxs[i][0]][0]
        buf = buf.reshape([1,buf.shape[1],buf.shape[2],buf.shape[0]])
        for k in range(3):
            buf[:,:,:,k] = data_loader.dataset[top_idxs[i][0]][0][k]
            buf[:,:,:,k] = (buf[:,:,:,k]-buf[:,:,:,k].min())/(buf[:,:,:,k].max()-buf[:,:,:,k].min())
        for j in range(1,top_k):
            buf2 = data_loader.dataset[top_idxs[i][j]][0]
            buf2 = buf2.reshape([1,buf2.shape[1],buf2.shape[2],buf2.shape[0]])
            for k in range(3):
                buf2[:,:,:,k] = data_loader.dataset[top_idxs[i][j]][0][k]
                buf2[:,:,:,k] = (buf2[:,:,:,k]-buf2[:,:,:,k].min())/(buf2[:,:,:,k].max()-buf2[:,:,:,k].min())
            buf = np.concatenate([buf,buf2],axis=0)
        top_examples[i] = buf

        #top_examples[i] = data_loader.dataset[top_idxs[i]][0]
        #top_examples[i] = top_examples[i].reshape([top_examples[i].shape[0],top_examples[i].shape[2],top_examples[i].shape[3]])
        #top_examples[i] = data_loader.dataset.test_data[top_idxs[i]]
    #top_examples =

    # Before, i was doing this manually :
        # for i in range(activations.shape[0]):
        #     #pdb.set_trace()
        #     for j in range(num_concepts):
        #         min_val  = top_activations[j].min()
        #         min_idx  = top_activations[j].argmin()
        #         if activations[i,j] >  min_val:
        #             # Put new one in place of min
        #             top_activations[j][min_idx]  = activations[i,j]
        #             top_examples[j][min_idx] = data[i, :, :, :].data.numpy().squeeze()
        #     #pdb.set_trace()
    # for k in range(num_concepts):
    #     #print(k)
    #     Z = [(v,e) for v,e in sorted(zip(top_activations[k],top_examples[k]),  key=lambda x: x[0], reverse = True)]
    #     top_activations[k], top_examples[k] = zip(*Z)

    if layout == 'horizontal':
        num_cols = top_k
        num_rows = num_concepts
        figsize=(num_cols, 1.2*num_rows)
    else:
        num_cols = num_concepts
        num_rows = top_k
        figsize=(1.4*num_cols, num_rows)

    fig, axes  = plt.subplots(figsize=figsize, nrows=num_rows, ncols=num_cols )

    for i in range(num_concepts):
        for j in range(top_k):
            pos = (i,j) if layout == 'horizontal' else (j,i)

            l = i*top_k + j
            #print(i,j)
            #print(top_examples[i][j].shape)
            #axes[pos].imshow(top_examples[i][j], cmap='Greys',  interpolation='nearest')
            axes[pos].imshow(top_examples[i][j], interpolation='nearest')
            if layout == 'vertical':
                axes[pos].axis('off')
                if j == 0:
                    axes[pos].set_title('Cpt {}'.format(i+1), fontsize = 24)
            else:
                axes[pos].set_xticklabels([])
                axes[pos].set_yticklabels([])
                axes[pos].set_yticks([])
                axes[pos].set_xticks([])
                for side in ['top', 'right', 'bottom', 'left']:
                    axes[i,j].spines[side].set_visible(False)
                if i == 0:
                    axes[pos].set_title('Proto {}'.format(j+1))
                if j == 0:
                    axes[pos].set_ylabel('Concept {}'.format(i+1), rotation = 90)

    print('Done')

    # cols = ['Prot.{}'.format(col) for col in range(1, num_cols + 1)]
    # rows = ['Concept # {}'.format(row) for row in range(1, num_rows + 1)]
    #
    # for ax, col in zip(axes[0], cols):
    #     ax.set_title(col)
    #
    # for ax, row in zip(axes[:,0], rows):
    #     ax.set_ylabel(row, rotation=0, size='large')
    #plt.tight_layout()

    if layout == 'vertical':
        fig.subplots_adjust(wspace=0.01, hspace=0.1)
    else:
        fig.subplots_adjust(wspace=0.1, hspace=0.01)

    if save_path is not None:
        plt.savefig(save_path, bbox_inches = 'tight', format='pdf', dpi=300)
    plt.show()
    if return_fig:
        return fig, axes


def plot_prob_drop(attribs, prob_drop, save_path = None):

    ind = np.arange(len(attribs))
    column_names = [str(j) for j in range(1,22)]

    width = 0.65

    fig, ax1 = plt.subplots(figsize=(8,4))

    color1 = '#377eb8'
    ax1.bar(ind+width+0.35, attribs, 0.45, color=color1)
    ax1.set_ylabel(r'Feature Relevance $\theta(x)_i$',color=color1, fontsize = 14)
    #ax1.set_ylim(-1,1)
    ax1.set_xlabel('Feature')
    ax1.tick_params(axis='y', colors=color1)


    color2 = '#ff7f00'
    ax2 = ax1.twinx()
    ax2.ticklabel_format(style='sci',scilimits=(-2,2),axis = 'y')
    ax2.plot(ind+width+0.35, prob_drop, 'bo', linestyle='dashed', color=color2)
    ax2.set_ylabel('Probability Drop', color = color2, fontsize = 14)
    ax2.tick_params(axis='y', colors=color2)


    ax1.set_xticks(ind+width+(width/2))
    ax1.set_xticklabels(column_names)

    fig.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches = 'tight', format='pdf', dpi=300)

    plt.show()

def noise_stability_plots(model, dataset, cuda, save_path):
    # find one example of each digit:
    examples = {}
    i = 0
    while (not len(examples.keys()) == 10) and (i < len(dataset)):
        if dataset[i][1] not in examples:
            examples[dataset[i][1]] = dataset[i][0].view(1,1,28,28)
        i += 1

    for i in range(10):
        x = Variable(examples[i], volatile = True)
        if cuda:
            x = x.cuda()
        plot_theta_stability(model, x, noise_level = 0.5,
                save_path=save_path + '/noise_stability_{}.pdf'.format(i))

def plot_digit(x, ax = None, greys = True):
    if ax is None:
        fig, ax = plt.subplots()
    if type(x) is torch.Tensor:
        x = x.numpy()
    if x.ndim > 2:
        x = x.squeeze()
    if greys:
        ax.imshow(x, cmap='Greys')
    else:
        ax.imshow(x)
    ax.axis('off')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
