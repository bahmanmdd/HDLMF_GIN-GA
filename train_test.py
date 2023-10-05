"""
   Created by: Bahman Madadi
   Description: train-test pipeline for learning to solve DUEs with GNN
   partially adapted from: https://github.com/graphdeeplearning/benchmarking-gnns/blob/master/main_TSP_edge_classification.py
"""

import dgl
import numpy as np
import pandas as pd
import os
import socket
import time
import random
import glob
import argparse, json
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm

import data_due_generate as dg
import data_dataset_prep as dp
from gnn_net_load import gnn_model
from gnn_train import train_epoch, evaluate_network
import visualize


# define problem, case study and model
def problem_spec():

    """
    The problem_spec function is used to specify the model, problem and network
    that will be used in the simulation. The function returns a tuple of strings
    containing these three values.

    :return: The model, problem and network
    """

    model = 'GIN'
    problem = 'DUE'
    network = 'SiouxFalls'

    return model, problem, network


def get_data_dirs(problem, network):

    """
    The get_data_dirs function takes in a problem and network name, and returns the output directory, data directory,
    and dataset address for that particular problem/network combination. The output directory is where all of the
    output files will be saved to (e.g., plots). The data directory is where all of the input files are located (e.g.,
    the .pkl file containing the dataset). Finally, the dataset address is simply a string containing both directories
    and then ending with /{network}.pkl.

    :param problem: Specify the dataset to be used
    :param network: Determine the name of the network
    :return: The output directory, data directory and dataset address
    """

    out_dir = f'output/{problem}/{network}/'
    data_dir = f'Datasets{problem}/{network}/'
    dataset_address = f'Datasets{problem}/{network}/{network}.pkl'

    return out_dir, data_dir, dataset_address


# set important parameters
def set_parameters(model_name, problem, network):
    """
    The set_parameters function is used to set the parameters for a given model.

    :param model_name: Determine which model to use
    :param problem: Determine the dataset to be used
    :param network: Determine which network to use
    :return: Dataset, parameters, device and input/output directories
    """

    out_dir, data_dir, dataset_address = get_data_dirs(problem, network)
    dataset = dp.DUEDataset(problem, network)

    with open(f'configs/{problem}_{model_name}_{network}.json') as f:
        config = json.load(f)

    device = gpu_setup(config['gpu']['use'], config['gpu']['id'])
    params_opt = config['params_opt']
    params_net = config['params_net']
    params_net['device'] = device
    params_net['in_dim'] = dataset.train[0][0].ndata['feat'][0].shape[0]
    params_net['in_dim_edge'] = dataset.train[0][0].edata['feat'][0].size(0)
    params_net['total_param'] = get_total_model_params(model_name, params_net)

    return dataset, params_opt, params_net, device, data_dir, out_dir


# view and check total parameters in the model
def get_total_model_params(MODEL_NAME, params_net):
    """
    The get_total_model_params function takes in the model name and a dictionary of parameters for that model.
    It then creates an instance of the model with those parameters, and returns the total number of trainable
    parameters in that instance.

    :param MODEL_NAME: Specify the model to be used
    :param params_net: Pass the parameters of the network to be used
    :return: The total number of parameters in the model
    """

    model = gnn_model(MODEL_NAME, params_net)
    total_param = 0
    # print("MODEL DETAILS:\n")
    # print(model)
    for param in model.parameters():
        # print(param.data.size())
        total_param += np.prod(list(param.data.size()))
    # print('MODEL/Total parameters:', MODEL_NAME, total_param)
    return total_param


# GPU Setup
def gpu_setup(use_gpu, gpu_id):
    """
    The gpu_setup function is used to set up the GPU for training.

    :param use_gpu: Determine whether to use gpu or not
    :param gpu_id: Specify which gpu to use
    :return: A device object
    """

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    if torch.cuda.is_available() and use_gpu:
        print('cuda available with GPU:', torch.cuda.get_device_name(0))
        device = torch.device("cuda")
    else:
        print('cuda not available')
        device = torch.device("cpu")
    return device


# prepare directories to save results
def prep_out_dirs(MODEL_NAME, DATASET_NAME, out_dir):

    """
    The prep_out_dirs function creates the following directories:
    1. A directory for saving model checkpoints (root_ckpt_dir)
    2. A directory for writing training logs to (root_log_dir)
    3. A directory for writing results files to (write_file_name)

    :param MODEL_NAME: Determine the model to be used
    :param DATASET_NAME: Determine the dataset
    :param out_dir: Specify the higher level directory for the logs, models, checkpoints and results
    :return: The directories
    """

    trial_name = DATASET_NAME + "_" + MODEL_NAME + "_" + time.strftime('%Y-%m-%d_%H%M%S')
    root_log_dir = out_dir + 'logs/' + trial_name
    root_model_dir = out_dir + 'models/' + trial_name
    root_ckpt_dir = out_dir + 'checkpoints/' + trial_name
    write_config_file = out_dir + 'configs/' + trial_name
    write_file_name = out_dir + 'results/' + trial_name + "/"
    dirs = root_log_dir, root_ckpt_dir, write_file_name, write_config_file, root_model_dir

    if not os.path.exists(out_dir + 'results'):
        os.makedirs(out_dir + 'results')

    if not os.path.exists(write_file_name):
        os.makedirs(write_file_name)

    if not os.path.exists(out_dir + 'configs'):
        os.makedirs(out_dir + 'configs')

    if not os.path.exists(out_dir + 'models'):
        os.makedirs(out_dir + 'models')

    if not os.path.exists(root_ckpt_dir):
        os.makedirs(root_ckpt_dir)

    return dirs


# TRAINING pipeline
def train_val_pipeline(MODEL_NAME, dataset, params_opt, params_net, dirs, device):
    """
    The train_val_pipeline function is the main function that trains and evaluates a model.

    :param MODEL_NAME: Specify which model to use
    :param dataset: Specify the dataset
    :param params_opt: Pass the hyperparameters for optimization
    :param params_net: Pass the parameters of the neural network to be trained
    :param dirs: Store the results of training
    :param device: Specify the device to use for training (cpu or gpu)
    """

    t0 = time.time()
    per_epoch_time = []

    DATASET_NAME = dataset.name
    trainset, valset, testset = dataset.train, dataset.val, dataset.test

    root_log_dir, root_ckpt_dir, file_name, write_config_file, root_model_dir = dirs

    # Write the network and optimization hyper-parameters in output config folder
    with open(write_config_file + '.txt', 'w') as f:
        f.write("""Dataset: {},\nModel: {}\n\nparams={}\n\nnet_params={}\n\n\nTotal Parameters: {}\n\n""".format
            (DATASET_NAME, MODEL_NAME, params_opt, params_net, params_net['total_param']))

    log_dir = os.path.join(root_log_dir, "RUN_" + str(0))
    writer = SummaryWriter(log_dir=log_dir)

    # setting seeds
    random.seed(params_opt['seed'])
    np.random.seed(params_opt['seed'])
    torch.manual_seed(params_opt['seed'])
    if device.type == 'cuda':
        torch.cuda.manual_seed(params_opt['seed'])

    print("Training Graphs: ", len(trainset))
    print("Validation Graphs: ", len(valset))
    print("Test Graphs: ", len(testset))

    model = gnn_model(MODEL_NAME, params_net)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=params_opt['init_lr'], weight_decay=params_opt['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                     factor=params_opt['lr_reduce_factor'],
                                                     patience=params_opt['lr_schedule_patience'],
                                                     verbose=True)

    epoch_train_losses, epoch_val_losses, epoch_test_losses = [], [], []
    epoch_train_mapes, epoch_val_mapes, epoch_test_mapes = [], [], []
    epoch_train_ttts, epoch_val_ttts, epoch_test_ttts = [], [], []
    epoch_train_r2s, epoch_val_r2s, epoch_test_r2s = [], [], []
    best_train_loss = np.inf
    best_train_accu = np.inf
    best_val_loss = np.inf
    best_val_accu = np.inf
    best_test_loss = np.inf
    best_test_accu = np.inf

    train_loader = DataLoader(trainset, batch_size=params_opt['batch_size'], shuffle=True, collate_fn=dataset.collate)
    val_loader = DataLoader(valset, batch_size=params_opt['batch_size'], shuffle=False, collate_fn=dataset.collate)
    test_loader = DataLoader(testset, batch_size=params_opt['batch_size'], shuffle=False, collate_fn=dataset.collate)

    # At any point you can hit Ctrl + C to break out of training early.
    print('----------------------------------------------------------------------------------------')
    print('Training in progress (at any point you can hit Ctrl + C to break out of training early)!')
    print('----------------------------------------------------------------------------------------')
    try:
        with tqdm(range(params_opt['epochs']), colour='green') as t:

            for epoch in t:

                t.set_description('Epoch %d' % epoch)

                start = time.time()

                epoch_train_loss, epoch_train_mape, epoch_train_ttt, epoch_train_r2, optimizer = train_epoch(model, optimizer, device, train_loader, epoch)
                epoch_val_loss, epoch_val_mape, epoch_val_ttt, epoch_val_r2, _, _ = evaluate_network(model, device, val_loader, epoch)
                epoch_test_loss, epoch_test_mape, epoch_test_ttt, epoch_test_r2, _, _ = evaluate_network(model, device, test_loader, epoch)

                epoch_train_losses.append(epoch_train_loss)
                epoch_train_mapes.append(epoch_train_mape)
                epoch_train_ttts.append(epoch_train_ttt)
                epoch_train_r2s.append(epoch_train_r2)

                epoch_val_losses.append(epoch_val_loss)
                epoch_val_mapes.append(epoch_val_mape)
                epoch_val_ttts.append(epoch_val_ttt)
                epoch_val_r2s.append(epoch_val_r2)

                epoch_test_losses.append(epoch_test_loss)
                epoch_test_mapes.append(epoch_test_mape)
                epoch_test_ttts.append(epoch_test_ttt)
                epoch_test_r2s.append(epoch_test_r2)

                # we update everything when loss improves
                # normally you do this based on validation loss but for CO train makes more sense
                if epoch_train_loss < best_train_loss:
                    best_val_loss = epoch_val_loss
                    best_val_accu = epoch_val_ttt
                    best_test_loss = epoch_test_loss
                    best_test_accu = epoch_test_ttt
                    best_train_loss = epoch_train_loss
                    best_train_accu = epoch_train_ttt

                    # checkpoint: save the model in model folder and in checkpoint
                    torch.save(model.state_dict(), f'{root_model_dir}.pt')
                    torch.save(model.state_dict(), f'{file_name}model_{DATASET_NAME}.pt')
                    torch.save(model.state_dict(), '{}.pkl'.format(root_ckpt_dir + "/model"))
                    # just for testing to make sure it is saved right and can be loaded
                    # model.load_state_dict(torch.load(f'{root_model_dir}.pt'))

                writer.add_scalar('train/_loss', epoch_train_loss, epoch)
                writer.add_scalar('train/_mape', epoch_train_mape, epoch)
                writer.add_scalar('train/_ttt', epoch_train_ttt, epoch)
                writer.add_scalar('train/_r2', epoch_train_r2, epoch)
                writer.add_scalar('val/_loss', epoch_val_loss, epoch)
                writer.add_scalar('val/_mape', epoch_val_mape, epoch)
                writer.add_scalar('val/_ttt', epoch_val_ttt, epoch)
                writer.add_scalar('val/_r2', epoch_val_r2, epoch)
                writer.add_scalar('test/_loss', epoch_test_loss, epoch)
                writer.add_scalar('test/_mape', epoch_test_mape, epoch)
                writer.add_scalar('test/_ttt', epoch_test_ttt, epoch)
                writer.add_scalar('test/_r2', epoch_test_r2, epoch)
                writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)

                t.set_postfix(
                    # time=time.time() - start,
                    lr=optimizer.param_groups[0]['lr'],
                    train_loss=epoch_train_loss,
                    val_loss=epoch_val_loss,
                    # test_loss=epoch_test_loss,
                    train_accu=1-epoch_train_ttt,
                    val_accu=1-epoch_val_ttt
                )

                per_epoch_time.append(time.time()-start)

                scheduler.step(epoch_val_loss)

                if optimizer.param_groups[0]['lr'] < params_opt['min_lr']:
                    print("\n!! LR EQUAL TO MIN LR SET.")
                    break

                # Stop training after params['max_time'] hours
                if time.time() - t0 > params_opt['max_time'] * 3600:
                    print('-' * 89)
                    print("Max_time for training elapsed {:.2f} hours, so stopping".format(params_opt['max_time']))
                    break

    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early because of KeyboardInterrupt')

    print('******************************')
    print('------------------------------')
    print('Based on best validation loss:')
    print('------------------------------')
    print("Best Val Loss: {:.4f}".format(best_val_loss))
    print("Best Test Loss: {:.4f}".format(best_test_loss))
    print("Best Train Loss: {:.4f}".format(best_train_loss))
    print('------------------------------')
    print("Best Val Accuracy: {:.4f}".format(1 - best_val_accu))
    print("Best Test Accuracy: {:.4f}".format(1 - best_test_accu))
    print("Best Train Accuracy: {:.4f}".format(1 - best_train_accu))
    print('------------------------------')
    print("Convergence Time (Epochs): {}".format(int(epoch+1)))
    print("Total Time: {:.2f} hours".format(float(time.time() - t0)/3600))
    print("AVG Epoch Time: {:.2f} seconds".format(np.mean(per_epoch_time)))
    print('------------------------------')
    print('Last Epoch results:')
    print('------------------------------')
    print("Val Accuracy: {:.4f}".format(1 - epoch_val_ttt))
    print("Test Accuracy: {:.4f}".format(1 - epoch_test_ttt))
    print("Train Accuracy: {:.4f}".format(1 - epoch_train_ttt))
    print('******************************')

    writer.close()

    test_loss, _, _, _, test_labels, test_scores = evaluate_network(model, device, test_loader, epoch)
    train_loss, _, _, _, _, _ = evaluate_network(model, device, train_loader, epoch)

    # write epoch KPIs to csv
    pd.DataFrame(torch.cat((test_labels, test_scores), 1).numpy(),
                 columns=['label', 'prediction']).to_csv(f'{file_name}Prediction-1batch.csv')
    pd.DataFrame(np.array([epoch_train_ttts,
                           epoch_test_ttts,
                           epoch_train_losses,
                           epoch_test_losses,
                           epoch_train_mapes,
                           epoch_test_mapes,
                           epoch_train_r2s,
                           epoch_test_r2s]).transpose(), columns=['train_ttt_error', 'test_ttt_error',
                                                                  'train_loss', 'test_loss',
                                                                  'train_mape', 'test_mape',
                                                                  'train_r2', 'test_r2']).to_csv(f'{file_name}KPI.csv',
                                                                                                 index_label='epoch')

    # save training visualization figures
    visualize.plot_results(epoch_train_losses, epoch_test_losses, 'Loss (MSE)', file_name)
    visualize.plot_results(epoch_train_mapes, epoch_test_mapes, 'Duality gap (MAPE)', file_name)
    visualize.plot_results(epoch_train_r2s, epoch_test_r2s, 'R squared', file_name)
    visualize.plot_results(epoch_train_ttts, epoch_test_ttts, 'Total travel time prediction error (MAPE)', file_name)
    visualize.plot_results(1 - np.array(epoch_train_ttts), 1 - np.array(epoch_test_ttts),
                           'Total travel time prediction accuracy (MAP)', file_name)

    # Write the results in out_dir/results folder
    with open(file_name + '_summary.txt', 'w') as f:
        f.write("""Dataset: {},\nModel: {}\n\nparams={}\n\nnet_params={}\n\n{}\n\nTotal Parameters: {}\n\n
    FINAL RESULTS\n
    TEST Loss (MSE): {:.4f}\n
    TRAIN Loss (MSE): {:.4f}\n
    TEST Accu (MAP): {:.4f}\n
    TRAIN Accu (MAP): {:.4f}\n\n
    Convergence Time (Epochs): {:.4f}\n
    Total Time Taken: {:.4f}hrs\n
    Average Time Per Epoch: {:.4f}s\n\n\n""" .format(
            DATASET_NAME, MODEL_NAME, params_opt, params_net, model, params_net['total_param'],
            np.mean(np.array(best_test_loss)), np.mean(np.array(best_train_loss)),
            np.mean(np.array(1 - best_test_accu)), np.mean(np.array(1 - best_train_accu)),
            epoch, (time.time()-t0)/3600, np.mean(per_epoch_time)))

    print('Done!')


def main():

    """
    The main function is the entry point of the training program.
    It calls all other functions in order to train and validate a model.
    """

    model_name, problem, network = problem_spec()

    dataset, params_opt, params_net, device, data_dir, out_dir = set_parameters(model_name, problem, network)

    dirs = prep_out_dirs(model_name, dataset.name, out_dir)

    train_val_pipeline(model_name, dataset, params_opt, params_net, dirs, device)


if __name__ == "__main__":
    main()

