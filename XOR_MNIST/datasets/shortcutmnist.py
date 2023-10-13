from datasets.utils.base_dataset import BaseDataset, get_loader
from datasets.utils.mnist_creation import load_2MNIST
from backbones.addmnist_joint import MNISTPairsEncoder, MNISTPairsDecoder
from backbones.addmnist_single import MNISTSingleEncoder
import numpy as np
from copy import deepcopy


class SHORTMNIST(BaseDataset):
    NAME = 'shortmnist'
    DATADIR='data/raw'

    def get_data_loaders(self):
        dataset_train, dataset_val, dataset_test  = load_2MNIST(c_sup=self.args.c_sup, 
                                                                which_c=self.args.which_c, 
                                                                args=self.args)

        print(dataset_train)

        ood_test = self.get_ood_test(dataset_test)

        self.filtrate(dataset_train, dataset_val, dataset_test)

        self.train_loader = get_loader(dataset_train, self.args.batch_size, val_test=False)
        self.val_loader   = get_loader(dataset_val,   self.args.batch_size, val_test=True)
        self.test_loader  = get_loader(dataset_test,  self.args.batch_size, val_test=True)
        self.ood_loader   = get_loader(ood_test,      self.args.batch_size, val_test=False)

        return self.train_loader, self.val_loader, self.test_loader

    def get_backbone(self):
        if self.args.joint:
            return MNISTPairsEncoder(),  MNISTPairsDecoder()
        else:
            return MNISTSingleEncoder(), MNISTPairsDecoder()
    
    def get_split(self):
        if self.args.joint:
            return 1, (10,10)
        else: 
            return 2, (10,)
        
    def filtrate(self, train_dataset, val_dataset, test_dataset):

        train_c_mask1 = ((train_dataset.real_concepts[:,0] == 0) & (train_dataset.real_concepts[:,1] == 6)) | \
                        ((train_dataset.real_concepts[:,0] == 2) & (train_dataset.real_concepts[:,1] == 8)) | \
                        ((train_dataset.real_concepts[:,0] == 4) & (train_dataset.real_concepts[:,1] == 6)) | \
                        ((train_dataset.real_concepts[:,0] == 4) & (train_dataset.real_concepts[:,1] == 8)) | \
                        ((train_dataset.real_concepts[:,0] == 1) & (train_dataset.real_concepts[:,1] == 5)) | \
                        ((train_dataset.real_concepts[:,0] == 3) & (train_dataset.real_concepts[:,1] == 7)) | \
                        ((train_dataset.real_concepts[:,0] == 1) & (train_dataset.real_concepts[:,1] == 9)) | \
                        ((train_dataset.real_concepts[:,0] == 3) & (train_dataset.real_concepts[:,1] == 9))  #| \
                        # ((train_dataset.real_concepts[:,0] == 5) & (train_dataset.real_concepts[:,1] == 7)) 
        train_c_mask2 = ((train_dataset.real_concepts[:,1] == 0) & (train_dataset.real_concepts[:,0] == 6)) | \
                        ((train_dataset.real_concepts[:,1] == 2) & (train_dataset.real_concepts[:,0] == 8)) | \
                        ((train_dataset.real_concepts[:,1] == 4) & (train_dataset.real_concepts[:,0] == 6)) | \
                        ((train_dataset.real_concepts[:,1] == 4) & (train_dataset.real_concepts[:,0] == 8)) | \
                        ((train_dataset.real_concepts[:,1] == 1) & (train_dataset.real_concepts[:,0] == 5)) | \
                        ((train_dataset.real_concepts[:,1] == 3) & (train_dataset.real_concepts[:,0] == 7)) | \
                        ((train_dataset.real_concepts[:,1] == 1) & (train_dataset.real_concepts[:,0] == 9)) | \
                        ((train_dataset.real_concepts[:,1] == 3) & (train_dataset.real_concepts[:,0] == 9)) # | \
                        # ((train_dataset.real_concepts[:,0] == 7) & (train_dataset.real_concepts[:,1] == 5)) 
        train_mask = np.logical_or(train_c_mask1, train_c_mask2)

        val_c_mask1 = ((val_dataset.real_concepts[:,0] == 0) & (val_dataset.real_concepts[:,1] == 6)) | \
                      ((val_dataset.real_concepts[:,0] == 2) & (val_dataset.real_concepts[:,1] == 8)) | \
                      ((val_dataset.real_concepts[:,0] == 4) & (val_dataset.real_concepts[:,1] == 6)) | \
                      ((val_dataset.real_concepts[:,0] == 4) & (val_dataset.real_concepts[:,1] == 8)) | \
                      ((val_dataset.real_concepts[:,0] == 1) & (val_dataset.real_concepts[:,1] == 5)) | \
                      ((val_dataset.real_concepts[:,0] == 3) & (val_dataset.real_concepts[:,1] == 7)) | \
                      ((val_dataset.real_concepts[:,0] == 1) & (val_dataset.real_concepts[:,1] == 9)) | \
                      ((val_dataset.real_concepts[:,0] == 3) & (val_dataset.real_concepts[:,1] == 9)) # | \
                    #   ((val_dataset.real_concepts[:,0] == 5) & (val_dataset.real_concepts[:,1] == 7)) 
        val_c_mask2 = ((val_dataset.real_concepts[:,1] == 0) & (val_dataset.real_concepts[:,0] == 6)) | \
                      ((val_dataset.real_concepts[:,1] == 2) & (val_dataset.real_concepts[:,0] == 8)) | \
                      ((val_dataset.real_concepts[:,1] == 4) & (val_dataset.real_concepts[:,0] == 6)) | \
                      ((val_dataset.real_concepts[:,1] == 4) & (val_dataset.real_concepts[:,0] == 8)) | \
                      ((val_dataset.real_concepts[:,1] == 1) & (val_dataset.real_concepts[:,0] == 5)) | \
                      ((val_dataset.real_concepts[:,1] == 3) & (val_dataset.real_concepts[:,0] == 7)) | \
                      ((val_dataset.real_concepts[:,1] == 1) & (val_dataset.real_concepts[:,0] == 9)) | \
                      ((val_dataset.real_concepts[:,1] == 3) & (val_dataset.real_concepts[:,0] == 9)) # | \
                    #   ((val_dataset.real_concepts[:,1] == 5) & (val_dataset.real_concepts[:,0] == 7)) 
        val_mask = np.logical_or(val_c_mask1, val_c_mask2)

        test_c_mask1 = ((test_dataset.real_concepts[:,0] == 0) & (test_dataset.real_concepts[:,1] == 6)) | \
                       ((test_dataset.real_concepts[:,0] == 2) & (test_dataset.real_concepts[:,1] == 8)) | \
                       ((test_dataset.real_concepts[:,0] == 4) & (test_dataset.real_concepts[:,1] == 6)) | \
                       ((test_dataset.real_concepts[:,0] == 4) & (test_dataset.real_concepts[:,1] == 8)) | \
                       ((test_dataset.real_concepts[:,0] == 1) & (test_dataset.real_concepts[:,1] == 5)) | \
                       ((test_dataset.real_concepts[:,0] == 3) & (test_dataset.real_concepts[:,1] == 7)) | \
                       ((test_dataset.real_concepts[:,0] == 1) & (test_dataset.real_concepts[:,1] == 9)) | \
                       ((test_dataset.real_concepts[:,0] == 3) & (test_dataset.real_concepts[:,1] == 9)) # | \
                    #    ((test_dataset.real_concepts[:,0] == 5) & (test_dataset.real_concepts[:,1] == 7)) 

        test_c_mask2 = ((test_dataset.real_concepts[:,1] == 0) & (test_dataset.real_concepts[:,0] == 6)) | \
                       ((test_dataset.real_concepts[:,1] == 2) & (test_dataset.real_concepts[:,0] == 8)) | \
                       ((test_dataset.real_concepts[:,1] == 4) & (test_dataset.real_concepts[:,0] == 6)) | \
                       ((test_dataset.real_concepts[:,1] == 4) & (test_dataset.real_concepts[:,0] == 8)) | \
                       ((test_dataset.real_concepts[:,1] == 1) & (test_dataset.real_concepts[:,0] == 5)) | \
                       ((test_dataset.real_concepts[:,1] == 3) & (test_dataset.real_concepts[:,0] == 7)) | \
                       ((test_dataset.real_concepts[:,1] == 1) & (test_dataset.real_concepts[:,0] == 9)) | \
                       ((test_dataset.real_concepts[:,1] == 3) & (test_dataset.real_concepts[:,0] == 9)) # | \
                    #    ((test_dataset.real_concepts[:,1] == 5) & (test_dataset.real_concepts[:,0] == 7)) 

        test_mask = np.logical_or(test_c_mask1, test_c_mask2)

        train_dataset.data = train_dataset.data[train_mask]
        val_dataset.data   = val_dataset.data[val_mask]  
        test_dataset.data  = test_dataset.data[test_mask]

        train_dataset.concepts = train_dataset.concepts[train_mask]
        val_dataset.concepts   = val_dataset.concepts[val_mask]
        test_dataset.concepts  = test_dataset.concepts[test_mask]

        train_dataset.targets = np.array(train_dataset.targets)[train_mask]
        val_dataset.targets   = np.array(val_dataset.targets)[val_mask] 
        test_dataset.targets  = np.array(test_dataset.targets)[test_mask]

    def get_ood_test(self, test_dataset):
        
        ood_test = deepcopy(test_dataset)

        test_c_mask1 = ((test_dataset.real_concepts[:,0] == 0) & (test_dataset.real_concepts[:,1] == 6)) | \
                       ((test_dataset.real_concepts[:,0] == 2) & (test_dataset.real_concepts[:,1] == 8)) | \
                       ((test_dataset.real_concepts[:,0] == 4) & (test_dataset.real_concepts[:,1] == 6)) | \
                       ((test_dataset.real_concepts[:,0] == 4) & (test_dataset.real_concepts[:,1] == 8)) | \
                       ((test_dataset.real_concepts[:,0] == 1) & (test_dataset.real_concepts[:,1] == 5)) | \
                       ((test_dataset.real_concepts[:,0] == 3) & (test_dataset.real_concepts[:,1] == 7)) | \
                       ((test_dataset.real_concepts[:,0] == 1) & (test_dataset.real_concepts[:,1] == 9)) | \
                       ((test_dataset.real_concepts[:,0] == 3) & (test_dataset.real_concepts[:,1] == 9))
        
        test_c_mask2 = ((test_dataset.real_concepts[:,1] == 0) & (test_dataset.real_concepts[:,0] == 6)) | \
                       ((test_dataset.real_concepts[:,1] == 2) & (test_dataset.real_concepts[:,0] == 8)) | \
                       ((test_dataset.real_concepts[:,1] == 4) & (test_dataset.real_concepts[:,0] == 6)) | \
                       ((test_dataset.real_concepts[:,1] == 4) & (test_dataset.real_concepts[:,0] == 8)) | \
                       ((test_dataset.real_concepts[:,1] == 1) & (test_dataset.real_concepts[:,0] == 5)) | \
                       ((test_dataset.real_concepts[:,1] == 3) & (test_dataset.real_concepts[:,0] == 7)) | \
                       ((test_dataset.real_concepts[:,1] == 1) & (test_dataset.real_concepts[:,0] == 9)) | \
                       ((test_dataset.real_concepts[:,1] == 3) & (test_dataset.real_concepts[:,0] == 9))
        test_mask = np.logical_and(~test_c_mask1, ~test_c_mask2)

        ood_test.data     = ood_test.data[test_mask]
        ood_test.concepts = ood_test.concepts[test_mask]
        ood_test.targets  = np.array(ood_test.targets)[test_mask]
        return ood_test

if __name__=='__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--c_sup', type=int, default=1)
    args = parser.parse_args()

    dataset = SHORTMNIST(args)
    dataset.get_data_loaders()