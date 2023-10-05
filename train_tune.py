"""
   Created by: Bahman Madadi
   Description: GNN (in this case GIN) hyperparameter tuning using hyperopt
"""

import os
import time
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval

from train_test import set_parameters
from gnn_train import train_epoch, evaluate_network
from gnn_net_load import gnn_model


# define problem, case study and model
def problem_spec():

    """
    The problem_spec function is used to specify the model, problem and network
    that will be used in the simulation. The function returns a tuple of strings
    containing these three values.

    :return: The following: model, problem and network
    """

    model = 'GIN'
    problem = 'DUE'
    network = 'SiouxFalls'

    return model, problem, network


def params_tune():

    """
    The params_tune function is used to tune the parameters of the model.
        The function returns two values: max_sample and max_epochs.
        These are used in the main function to determine how many samples and epochs will be run.

    :return: Two values, max_sample and max_epochs
    """

    max_sample = 27
    max_epochs = 30

    return max_sample, max_epochs


def choices_ho():

    """
    The choices_ho function is used to define the hyperparameter space for a
    hyperopt search. It returns a dictionary of all the parameters that will be
    searched over, and their possible values. The keys are strings, and the values
    are lists of possible choices for each parameter.

    :return: A dictionary with the hyperparameters to be tuned
    :doc-author: Trelent
    """

    space = {
        'L': hp.choice('L', [5, 7, 9]),
        # 'hidden_dim': hp.choice('hidden_dim', [75]),
        # 'residual': hp.choice('residual', [True, False]),
        # 'readout': hp.choice('readout', ["sum", "mean"]),
        'n_mlp_GIN': hp.choice('n_mlp_GIN', [3, 5, 7]),
        # 'learn_eps_GIN': hp.choice('learn_eps_GIN', [True, False]),
        # 'neighbor_aggr_GIN': hp.choice('neighbor_aggr_GIN', ["sum", "mean"]),
        # 'batch_norm': hp.choice('batch_norm', [True, False]),

        # 'batch_size': hp.choice('batch_size', [32, 64, 128]),
        # 'init_lr': hp.choice('init_lr', [0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]),
        # 'init_lr': hp.choice('init_lr', [0.05, 0.01, 0.005, 0.001]),
        'init_lr': hp.choice('init_lr', [0.001, 0.005, 0.01]),
        # 'lr_reduce_factor': hp.choice('lr_reduce_factor', [0.4, 0.6, 0.8, 0.9])
    }

    return space


def prep_model():

    """
    The prep_model function is used to set up the model for training.

    :return: The model name, dataset, optimization parameters, network parameters and the directory where we will save the tuning results
    :doc-author: Trelent
    """

    model_name, problem, network = problem_spec()

    dataset, params_opt, params_net, device, data_dir, out_dir = set_parameters(model_name, problem, network)

    tune_dir = out_dir + 'tune'
    if not os.path.exists(tune_dir):
        os.makedirs(tune_dir)

    return model_name, dataset, params_opt, params_net, tune_dir, device


def train_tune(model_name, dataset, params, net_params, max_num_epochs, device):

    """
    The train_tune function is used to train and tune the model.
        Args:
            model_name (str): The name of the GNN model to be trained.
            dataset (Dataset): The Dataset object containing all data for training, validation, and testing.
            params (dict): A dictionary of parameters for training the GNN models. These include hyperparameters such as
                learning rate, weight decay, etc., as well as values specific to each type of GNN architecture such as
                number of layers in a GCN or number of heads in an Attention layer. See READ

    :param model_name: Specify the type of gnn model to be used
    :param dataset: Load the data
    :param params: Pass the parameters of the model to be trained
    :param net_params: Pass the parameters of the neural network to be trained
    :param max_num_epochs: Specify the maximum number of epochs for which we want to train our model
    :param device: Specify whether to use the cpu or gpu
    :return: The validation loss, the validation accuracy, the training loss and the training accuracy
    """

    trainset, valset = dataset.train, dataset.val
    model_gnn = gnn_model(model_name, net_params)
    model = model_gnn.to(device)

    optimizer = optim.Adam(model.parameters(), lr=params['init_lr'], weight_decay=params['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                     factor=params['lr_reduce_factor'],
                                                     patience=params['lr_schedule_patience'],
                                                     verbose=True)

    train_loader = DataLoader(trainset, batch_size=params['batch_size'], shuffle=True, collate_fn=dataset.collate)
    val_loader = DataLoader(valset, batch_size=params['batch_size'], shuffle=False, collate_fn=dataset.collate)

    for epoch in range(max_num_epochs):

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        loss, mape, ttt, _, optimizer = train_epoch(model, optimizer, device, train_loader, epoch)
        val_loss, val_mape, val_ttt, _, _, _ = evaluate_network(model, device, val_loader, epoch)

        scheduler.step(val_loss)

        # print statistics
        if np.mod(epoch, 10) == 0:
            print(f'epoch: {epoch}')
        # print(f'epoch: {epoch}')
        # print(f'train loss: {loss}')
        # print(f'val loss: {val_loss}')
        # print(f'val accuracy: {val_ttt}')

    return val_loss, val_ttt, loss, ttt


def fitness_bays(model_name, dataset, opt_params, net_params, max_num_epochs, device):

    """
    The fitness_bays function is a wrapper for the train_tune function.
    It takes as input the model name, dataset, optimization parameters and network parameters.
    The fitness_bays function then creates a new set of hyperparameters by sampling from the
    Bayesian Optimization space defined in bayes_opt(). The sampled hyperparameters are passed to
    the train_tune() function which trains and tunes a GNN on them. The validation loss is returned
    to BayesianOptimization which uses it to update its internal model of the objective function.

    :param model_name: Select the model to be trained
    :param dataset: Specify the dataset to be used
    :param opt_params: Pass the hyperparameters for optimization
    :param net_params: Pass the network parameters
    :param max_num_epochs: Set the maximum number of epochs for training
    :param device: Determine whether to run on the cpu or gpu
    :return: A function that can be called by the bayesianoptimizer
    """

    def fit(params):
        global best_loss
        global best_val_loss
        global best_accuracy
        global best_val_accuracy
        global best_params
        t0 = time.time()
        print('------------------')
        print(f'running new sample')

        params_temp_net = net_params.copy()
        params_temp_opt = opt_params.copy()

        for par in params:
            if par in params_temp_opt:
                params_temp_opt[par] = params[par]
            if par in params_temp_net:
                params_temp_net[par] = params[par]

        val_loss, val_ttt, loss, ttt = train_tune(model_name, dataset, params_temp_opt, params_temp_net, max_num_epochs, device)

        if loss < best_loss:
            best_loss = loss
            best_val_loss = val_loss
            best_accuracy = ttt
            best_val_accuracy = val_ttt
            best_params_opt = params_temp_opt.copy()
            best_params_gnn = params_temp_net.copy()
            best_params = params.copy()

            print('improved hyperparameter set:')
            print(best_params_opt)
            print(best_params_gnn)
            print(f'validation loss: {val_loss}')
            print(f'training loss: {loss}')
            print(f'validation accuracy: {1 - val_ttt}')
            print(f'training accuracy: {1 - ttt}')
        sample_time = time.time() - t0
        print(f'sample time: {sample_time} seconds')

        return {'loss': loss, 'status': STATUS_OK}
    return fit


def tune_bays(num_samples=50, max_num_epochs=10):

    """
    The tune_bays function is the main function that will be called to tune a model.
    It takes in two arguments: num_samples and max_num_epochs. The first argument, num_samples,
    is the number of samples that we want to run through our Bayesian optimization algorithm.
    The second argument, max_num epochs, is the maximum number of epochs we want each sample to train for before it's
    evaluated on validation data.

    :param num_samples: Specify the number of times you want to run your model
    :param max_num_epochs: Set the maximum number of epochs to train for
    :return: The best optimized parameters
    """

    global best_loss
    global best_val_loss
    global best_accuracy
    global best_val_accuracy
    global best_params

    model_name, dataset, opt_params, net_params, tune_dir, device = prep_model()
    space = choices_ho()

    trials = Trials()

    best = fmin(fitness_bays(model_name, dataset, opt_params, net_params, max_num_epochs, device), space,
                algo=tpe.suggest, max_evals=num_samples, trials=trials)

    with open(f'{tune_dir}/best_params.txt', 'w') as f:
        f.write("""Dataset: {}\nModel: {}\n\nbest tuned parameters\n\n\nTotal Parameters: {}\n\n""".format
                (dataset.name, model_name, best_params, net_params['total_param']))

    print('******************************************')
    print('******************************************')
    print(f'Best validation loss: {best_val_loss}')
    print(f'Best training loss: {best_loss}')
    print(f'Best validation accuracy: {1 - best_val_accuracy}')
    print(f'Best training accuracy: {1 - best_accuracy}')
    print('Best optimized parameters:')
    print(best_params)
    print('Done!')

    return best


def main():

    """
    The main function is the entry point of the tuning program.
    It calls all other functions in order to train and test a model,
    and then save it for later use.

    Returns: A dictionary with the best parameters
    """

    best_params = {}
    best_loss = np.inf
    best_val_loss = np.inf
    best_accuracy = 0
    best_val_accuracy = 0
    max_samples, max_epochs = params_tune()

    t0 = time.time()

    best_params_bays = tune_bays(num_samples=max_samples, max_num_epochs=max_epochs)

    total_time = time.time() - t0
    print(f'total time: {total_time}')


if __name__ == "__main__":

    main()


