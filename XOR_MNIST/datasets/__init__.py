
import os
import inspect
import importlib
from argparse import Namespace

def get_all_datasets():
    return [data.split('.')[0] for data in os.listdir('datasets')
            if not data.find('__') > -1 and 'py' in data]

NAMES = {}
for dataset in get_all_datasets():
    dat = importlib.import_module('datasets.' + dataset)

    dataset_classes_name = [x for x in dat.__dir__() if 'type' in str(type(getattr(dat, x))) \
                            and 'BaseDataset' in str(inspect.getmro(getattr(dat, x))[1:])]
    for d in dataset_classes_name:
        c = getattr(dat, d)
        NAMES[c.NAME] = c

def get_dataset(args: Namespace):
    """
    Creates and returns a continual dataset.
    :param args: the arguments which contains the hyperparameters
    :return: the continual dataset
    """
    assert args.dataset in NAMES.keys(), NAMES
    return NAMES[args.dataset](args)
