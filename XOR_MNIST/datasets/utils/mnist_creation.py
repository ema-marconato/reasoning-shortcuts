import os
from itertools import product
from pathlib import Path
from random import sample, choice
from torchvision import transforms
import numpy as np
import torch
from torch import Tensor, load
from torch.utils.data import Dataset
from torchvision import datasets
from tqdm import tqdm
import copy, itertools

def get_label(c1, c2, labels, args):
    if args.task == 'addition':
        return labels

    elif args.task == 'product' and not args.model in ['mnistltn', 'mnistltnrec']:
        n_queries = [0]
        for i,j in itertools.product(range(1,10), range(1,10)):
            n_queries.append(i*j)
        n_queries = np.unique(np.array(n_queries))

        prod = c1 * c2
        q = [np.argmax( np.array(n_queries == p, dtype=int) ) for p in prod]
        q = np.array(q)
        
        return q
    
    elif args.task == 'product' and args.model in ['mnistltn', 'mnistltnrec']:
        return c1*c2
    
    elif args.task == 'multiop' and not args.model in ['mnistltn', 'mnistltnrec']:
        multiop = np.vstack((c1+c2, c1*c2)).T
        
        ls = []
        for c in multiop:
            lif1 = np.sum(c - np.array([1,0])) == 0
            lif2 = np.sum(c - np.array([2,0])) == 0
            lif3 = np.sum(c - np.array([4,3])) == 0
            
            if   lif1: ls.append(0)
            elif lif2: ls.append(1)
            elif lif3: ls.append(2)
            else: ls.append(3)
        return np.array(ls)
    
    elif args.task == 'multiop' and args.model in ['mnistltn', 'mnistltnrec']:
        multiop = c1**2 + c2**2 + c1*c2
        return multiop

class nMNIST(Dataset):
    """nMNIST dataset."""

    def __init__(self, split:str, data_path, args):
        print(f'Loading {split} data')
        self.data, self.labels = self.read_data(path=data_path, split=split)
        self.targets= self.labels[:,-1:].reshape(-1)
        self.concepts= self.labels[:,:-1]
        self.real_concepts = np.copy(self.labels[:,:-1])

        self.targets = get_label(self.real_concepts[:,0], self.real_concepts[:,1], self.targets, args)

        normalize = transforms.Normalize((0.1307,), (0.3081,))

        self.transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),  # implicitly divides by 255
        # normalize
        ])

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
            image = self.data[idx]
            label=self.targets[idx]
            concepts=self.concepts[idx]

            if self.transform is not None:
                newimage=self.transform(image.astype("uint8"))

            return newimage, label,concepts

    def read_data(self, path, split):
        """
        Returns images and labels
        """
        try:
            # print("Loading data...")
            data = load(path)
            # print("Loaded.")
        except:
            print("No dataset found.")

    
        images = data[split]['images']
        labels = data[split]['labels']
        
        return images, labels

    def reset_counter(self):
        self.world_counter = {c: self.samples_x_world // self.batch_size for c in self.worlds}
        # self.world_counter = {c: self.samples_x_world for c in self.worlds}

    # def __copy__():

def create_sample(X, target_sequence, digit2idx):
    idxs = [choice(digit2idx[digit][0]) for digit in target_sequence]
    imgs = [X[idx] for idx in idxs]
    new_image = np.concatenate(imgs, axis=1)
    new_label = target_sequence + (np.sum(target_sequence),)

    return new_image, np.array(new_label).astype('int32'), idxs


def create_dataset(n_digit=2, sequence_len=2, samples_x_world=100, train=True, download=False):
    # Download data
    MNIST = datasets.MNIST(root='./data/raw/', train=train, download=download)

    x, y = MNIST.data, MNIST.targets

    # Create dictionary of indexes for each digit
    digit2idx = {k: [] for k in range(10)}
    for k, v in digit2idx.items():
        v.append(np.where(y == k)[0])

    # Create the list of all possible permutations with repetition of 'sequence_len' digits
    worlds = list(product(range(n_digit), repeat=sequence_len))
    imgs = []
    labels = []

    # Create data sample for each class
    for c in tqdm(worlds):
        for i in range(samples_x_world):
            img, label, idxs = create_sample(x, c, digit2idx)
            imgs.append(img)
            labels.append(label)

    # Create dictionary of indexes for each world
    label2idx = {c: set() for c in worlds}
    for k, v in tqdm(label2idx.items()):
        for i, label in enumerate(labels):
            if tuple(label[:sequence_len]) == k:
                v.add(i)
    label2idx = {k: torch.tensor(list(v)) for k, v in label2idx.items()}

    return np.array(imgs).astype('int32'), np.array(labels), label2idx


def check_dataset(n_digits, data_folder, data_file, dataset_dim, sequence_len=2):
    """Checks whether the dataset exists, if not creates it."""
    Path(data_folder).mkdir(parents=True, exist_ok=True)
    data_path = os.path.join(data_folder, data_file)
    try:
        load(data_path)
    except:
        print("No dataset found.")
        # Define dataset dimension so to have teh same number of worlds
        n_worlds = n_digits ** sequence_len
        samples_x_world = {k: int(d / n_worlds) for k, d in dataset_dim.items()}
        dataset_dim = {k: s * n_worlds for k, s in samples_x_world.items()}

        train_imgs, train_labels, train_indexes = create_dataset(n_digit=n_digits, sequence_len=sequence_len,
                                                                 samples_x_world=samples_x_world['train'], train=True,
                                                                 download=True)

        val_imgs, val_labels, val_indexes = create_dataset(n_digit=n_digits, sequence_len=sequence_len,
                                                           samples_x_world=samples_x_world['val'], train=True,
                                                           download=True)
        test_imgs, test_labels, test_indexes = create_dataset(n_digit=n_digits, sequence_len=sequence_len,
                                                              samples_x_world=samples_x_world['test'], train=False,
                                                              download=True)

        print(f"Dataset dimensions: \n\t{dataset_dim['train']} train ({samples_x_world['train']} samples per world), \n\t{dataset_dim['val']} validation ({samples_x_world['val']} samples per world), \n\t{dataset_dim['test']} test ({samples_x_world['test']} samples per world)")

        data = {'train': {'images': train_imgs, 'labels': train_labels},
                'val': {'images': val_imgs, 'labels': val_labels},
                'test': {'images': test_imgs, 'labels': test_labels}}

        indexes = {'train': train_indexes,
                   'val': val_indexes,
                   'test': test_indexes}

        torch.save(data, data_path)
        for key, value in indexes.items():
            torch.save(value, os.path.join(data_folder, f'{key}_indexes.pt'))

        print(f"Dataset saved in {data_folder}")

def load_2MNIST(n_digits=10, dataset_dimensions= {'train': 42000, 'val': 12000, 'test': 6000},
                c_sup=1, which_c=[-1], args=None):
    # Load data
    data_folder = os.path.dirname(os.path.abspath(__file__))
    data_folder = os.path.join(data_folder, f'2mnist_{n_digits}digits')
    data_file =  f'2mnist_{n_digits}digits.pt'

    # Check whether dataset exists, if not build it
    check_dataset(n_digits, data_folder, data_file, dataset_dimensions)
    train_set, val_set, test_set = load_data(data_file=data_file, data_folder=data_folder, 
                                             c_sup=c_sup, which_c=which_c,
                                             args=args)

    return train_set, val_set, test_set


def load_data(data_file, data_folder, c_sup=1, which_c=[-1], args=None):
    
    # Prepare data
    data_path = os.path.join(data_folder, data_file)
    train_set = nMNIST('train', data_path=data_path, args=args)
    val_set   = nMNIST('val',   data_path=data_path, args=args)
    test_set  = nMNIST('test',  data_path=data_path, args=args)

    r_seq = np.load('data/rn.npy')

    for i in range(len(train_set)):
        if r_seq[i] > c_sup:
            train_set.concepts[i] = -1
        elif not (which_c[0] == -1):
            if train_set.concepts[i,0] not in which_c:
                train_set.concepts[i,0] = -1
            if train_set.concepts[i,1] not in which_c:
                train_set.concepts[i,1] = -1
                
    return train_set, val_set, test_set
