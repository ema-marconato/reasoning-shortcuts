import os.path
import random
from datetime import datetime
from itertools import product
from math import isnan
from pathlib import Path
from time import time, sleep
import numpy as np
import pandas as pd
import torch
from problog.formula import LogicFormula, LogicDAG
from problog.sdd_formula import SDD
from torch import nn
import itertools

from problog.logic import Term, Constant
from problog.logic import Var, AnnotatedDisjunction


def lock_resource(lock_filename):
    with open(lock_filename, 'w') as f:
        f.write('locked')


def release_lock(lock_filename):
    os.remove(lock_filename)


def create_facts(sequence_len, n_digits=10):
    """
    Return the list of ADs necessary to describe an image with 'sequence_len' digits.
    'n_facts' specifies how many digits we are considering (i.e. n_facts = 2 means that the images can contain only 0 or 1)
    """

    ad = []  # Empty list to store the ADs
    for i in range(sequence_len):
        pos = i + 1
        annot_disj = ""  # Empty string to store the current AD facts

        # Build the AD
        digit = Term('digit')
        X = Var('X')
        facts = [digit(X, Constant(pos), Constant(y), p='p_' + str(pos) + str(y)) for y in range(n_digits)]
        annot_disj += str(AnnotatedDisjunction(facts, None)) + '.'

        ad.append(annot_disj)

    return ad


def define_ProbLog_model(facts, rules, label, digit_query=None, mode='query'):
    """Build the ProbLog model using teh given facts, rules, evidence and query."""
    model = ""  # Empty program

    # Insert annotated disjuctions
    for i in range(len(facts)):
        model += "\n\n% Digit in position " + str(i + 1) + "\n\n"
        model += facts[i]

    # Insert rules
    model += "\n\n% Rules\n"
    model += rules

    # Insert digit query
    if digit_query:
        model += "\n\n% Digit Query\n"
        model += "query(" + digit_query + ")."

    # Insert addition query
    if mode == 'query':
        model += "\n\n% Addition Query\n"
        model += "query(addition(img," + str(label) + "))."

    elif mode == 'evidence':
        model += "\n\n% Addition Evidence\n"
        model += "evidence(addition(img," + str(label) + "))."

    return model


def update_resource(log_filepath, update_info, lock_filename='access.lock'):
    # {'Experiment_ID': 0, 'Run_ID': 1, ...}
    print('Updating resource with: {}'.format(update_info))

    # Check if lock file does exist
    # If it exists -> I have to wait (sleep -> 1.0 second)
    while os.path.isfile(lock_filename):
        sleep(1.0)

    # Do lock
    lock_resource(lock_filename)

    # Do update
    try:
        log_file = open(log_filepath, 'a')
        log_file.write(update_info)
        log_file.close()
    except Exception as e:
        raise e
    finally:
        # Release lock
        release_lock(lock_filename)


def load_mnist_classifier(checkpoint_path, device):
    input_size = 784
    hidden_sizes = [128, 64]
    output_size = 10

    # Build a feed-forward network
    clf = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                        nn.ReLU(),
                        nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                        nn.ReLU(),
                        nn.Linear(hidden_sizes[1], output_size),
                        nn.LogSoftmax(dim=1))

    if torch.cuda.is_available():
        clf.load_state_dict(torch.load(checkpoint_path))
        clf = clf.to(device)
    else:
        clf.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu')))

    return clf


def define_experiment(exp_folder, exp_class, params, exp_counter):
    log_file = Path(os.path.join(exp_folder, exp_class, exp_class + '.csv'))
    params_columns = ['latent_dim_sub', 'latent_dim_sym', 'learning_rate', 'dropout', 'dropout_ENC', 'dropout_DEC',
                      'recon_w',
                      'kl_w',
                      'query_w', 'sup_w']
    if log_file.is_file():
        # Load file
        log_csv = pd.read_csv(os.path.join(exp_folder, exp_class, exp_class + '.csv'))

        # Check if the required number of test has been already satisfied
        required_exp = params['n_exp']

        if len(log_csv) > 0:
            query = ''.join(f' {key} == {params[key]} &' for key in params_columns)[:-1]
            n_exp = len(log_csv.query(query))
            if n_exp == 0:
                exp_ID = log_csv['exp_ID'].max() + 1
                if isnan(exp_ID):
                    exp_ID = 1
                counter = required_exp - n_exp
                print("\n\n{} compatible experiments found in file {} -> {} experiments to run.".format(n_exp,
                                                                                                        os.path.join(
                                                                                                            exp_folder,
                                                                                                            exp_class,
                                                                                                            exp_class + '.csv'),
                                                                                                        counter))

                run_ID = datetime.today().strftime('%d-%m-%Y-%H-%M-%S')
            elif n_exp < required_exp:
                exp_ID = log_csv.query(query)['exp_ID'].values[0]
                counter = required_exp - n_exp
                print("\n\n{} compatible experiments found in file {} -> {} experiments to run.".format(n_exp,
                                                                                                        os.path.join(
                                                                                                            exp_folder,
                                                                                                            exp_class,
                                                                                                            exp_class + '.csv'),
                                                                                                        counter))

                run_ID = datetime.today().strftime('%d-%m-%Y-%H-%M-%S')

            else:
                print("\n\n{} compatible experiments found in file {} -> No experiments to run.".format(n_exp,
                                                                                                        os.path.join(
                                                                                                            exp_folder,
                                                                                                            exp_class,
                                                                                                            exp_class + '.csv'),
                                                                                                        0))
                counter = 0
                exp_ID = log_csv.query(query)['exp_ID'].values[0]
                run_ID = None
        else:
            counter = required_exp
            exp_ID = log_csv['exp_ID'].max() + 1
            if isnan(exp_ID):
                exp_ID = 1
            run_ID = datetime.today().strftime('%d-%m-%Y-%H-%M-%S')
            print("\n\n0 compatible experiments found in file {} -> {} experiments to run.".format(
                exp_folder + exp_class + '.csv',
                counter))


    else:
        counter = params['n_exp']
        # Create log file
        log_file = open(os.path.join(exp_folder, exp_class, exp_class + '.csv'), 'w')
        header = 'exp_ID,run_ID,' + ''.join(str(key) + ',' for key in params_columns) + params[
            'rec_loss'] + "_recon_val,acc_discr_val," + params[
                     'rec_loss'] + "_recon_test,acc_discr_test,acc_gen,epochs,max_epoch,time,tag\n"
        log_file.write(header)
        # Define experiment ID
        exp_ID = 1
        run_ID = datetime.today().strftime('%d-%m-%Y-%H-%M-%S')
        print()
        print('-' * 40)
        print("\nNO csv file found -> new file created {}".format(
            os.path.join(exp_folder, exp_class, exp_class + '.csv')))
        print('-' * 40)
        print()
        log_file.close()

    exp_counter += 1
    print()
    print('*' * 40)
    print("Running exp {} (exp ID: {})".format(exp_counter, exp_ID))
    print("Parameters:", params)
    print('*' * 40)
    print()

    return run_ID, str(exp_ID), exp_counter, counter, params_columns


def build_model_dict(sequence_len, n_digits):
    """Define dictionary of pre-compiled ProbLog models"""
    possible_query_add = {2: list(range(0, (n_digits - 1) * 2 + 1))}
    rules = "addition(X,N) :- digit(X,1,N1), digit(X,2,N2), N is N1 + N2.\ndigits(X,Y):-digit(img,1,X), digit(img,2,Y)."
    facts = create_facts(sequence_len, n_digits=n_digits)
    model_dict = {'query': {add: "EMPTY" for add in possible_query_add[sequence_len]},
                  'evidence': {add: "EMPTY" for add in possible_query_add[sequence_len]}}

    for mode in ['query', 'evidence']:
        for add in model_dict[mode]:
            problog_model = define_ProbLog_model(facts, rules, label=add, digit_query='digits(X,Y)', mode=mode)
            lf = LogicFormula.create_from(problog_model)
            dag = LogicDAG.create_from(lf)
            sdd = SDD.create_from(dag)
            model_dict[mode][add] = sdd

    return model_dict


def build_worlds_queries_matrix(sequence_len=0, n_digits=0, task='addmnist'):
    """Build Worlds-Queries matrix"""
    if task == 'addmnist':
        possible_worlds = list(product(range(n_digits), repeat=sequence_len))
        n_worlds = len(possible_worlds)
        n_queries = len(range(0, 10 + 10))
        look_up = {i: c for i, c in zip(range(n_worlds), possible_worlds)}
        w_q = torch.zeros(n_worlds, n_queries)  # (100, 20)
        for w in range(n_worlds):
            digit1, digit2 = look_up[w]
            for q in range(n_queries):
                if digit1 + digit2 == q:
                    w_q[w, q] = 1
        return w_q
    
    elif task == 'productmnist':
        possible_worlds = list(product(range(n_digits), repeat=sequence_len))
        n_worlds  = len(possible_worlds)
        n_queries = [0]
        for i,j in itertools.product(range(1,10), range(1,10)):
            n_queries.append(i*j)
        n_queries = np.unique(np.array(n_queries))
        
        look_up = {i: c for i, c in zip(range(n_worlds), possible_worlds)}
        w_q = torch.zeros(n_worlds, len(n_queries))  # (100, boh)
        for w in range(n_worlds):
            digit1, digit2 = look_up[w]
            for i, q in enumerate(n_queries):
                if digit1 * digit2 == q:
                    w_q[w, i] = 1

        return w_q
    
    elif task == 'multiopmnist':
        possible_worlds = list(product(range(n_digits), repeat=sequence_len))
        n_worlds  = len(possible_worlds)
        n_queries = np.array([0,1,2,3])
        
        w_q = torch.zeros(n_worlds, len(n_queries))  # (16, 4)
        look_up = {i: c for i, c in zip(range(n_worlds), possible_worlds)}
        for w in range(n_worlds):
            digit1, digit2 = look_up[w]
            for i, q in enumerate(n_queries):
                if digit1 + digit2 == 1 and digit1*digit2 == 0:
                    w_q[w, 0] = 1
                elif digit1 + digit2 == 2 and digit1*digit2 == 0:
                    w_q[w, 1] = 1
                elif digit1 + digit2 == 4 and digit1*digit2 == 3:
                    w_q[w, 2] = 1
                else:
                    w_q[w, 3] = 1
        return w_q

    else:
        NotImplementedError('Wrong choice')

