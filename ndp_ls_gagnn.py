"""
   Created by: Bahman Madadi
   Description: generate an instance of D-NDP with lane swaps and solve using GA-GNN algorithm (Madadi 2023):
   https://arxiv.org/abs/2303.06024
"""

import itertools
import copy
import os
import sys
import time
import dgl
import numpy as np
import pandas as pd
import torch
from geneticalgorithm2 import geneticalgorithm2 as ga

import data_due_generate as dg
from gnn_net_load import gnn_model
from train_test import set_parameters


def parameters_gagnn():

    """
    The parameters_gagnn function is used to set the parameters for the genetic algorithm.
        The main parameters are:
            max_num_iteration - maximum number of iterations (generations) that can be performed by GA;
            population_size - size of population in each generation;
            mutation_probability - probability of mutation for each gene in a chromosome;
            elit_ratio - ratio between elite and non-elite chromosomes in a new generation (0 &lt; elit &lt; 1);
            parents_portion - ratio between parents and offspring in a new generation (0 &lt; parents_portion &lt; 1);
            crossover_type - 'one_point', 'two_point', 'uniform', 'segment', 'shuffle';
            mutation_type - 'uniform_by_center';
            selection_type - 'roulette', 'stochastic', 'sigma_scaling', 'ranking', 'linear_ranking', 'tournament';
            max_iteration_without_improv - maximum number of iterations without improvement

    :return: A dictionary with the parameters of the genetic algorithm
    """

    ga_gnn = {}
    ga_gnn['params'] = {'max_num_iteration': 10000000,
                        'population_size': 128,
                        'mutation_probability': 0.02,
                        'elit_ratio': 0.4,
                        'parents_portion': 0.5,
                        'crossover_type': 'uniform',   # 'one_point', 'two_point', 'uniform', 'segment', 'shuffle'
                        'mutation_type': 'uniform_by_center',
                        'selection_type': 'ranking',   # 'roulette', 'stochastic', 'sigma_scaling', 'ranking', 'linear_ranking', 'tournament'
                        'max_iteration_without_improv': 10000}
    ga_gnn['batch'] = True
    ga_gnn['top'] = 20
    ga_gnn['stud'] = True
    ga_gnn['revl'] = None  # start revolution after x generations of stagnation
    ga_gnn['dup_rmv'] = 1  # remove duplicates after x generations

    ga_gnn['no_plot'] = False
    ga_gnn['prg_bar'] = None  # 'stdout'

    return ga_gnn


def params_for_test():

    """
    The params_for_test function is used to set up the parameters for a test run of the model.
    It returns:
        network (str): The name of the network being tested.
        model (str): The type of graph neural net being used in this test run.  Options are 'GIN', 'GCN', and 'GraphSAGE'.
        problem (str): The type of problem that is being solved with this test run.  Options are 'DUE' and 'NDP_LA'.
            DUE stands for deterministic user equilibrium, while NDP_LA stands for non-deterministic path link
    """

    model = 'GIN'
    problem = 'DUE'
    networks = ['Anaheim']
    data_dir = 'TransportationNetworks'
    variation = {}
    variation['seed'] = 0
    variation['problem'] = 'NDP_LS'
    variation['swap_pr'] = 0.25
    variation['n_lanes'] = 2  # assumed number of existing lanes per link
    variation['max_lanes'] = 1  # max number of new lanes to add to each link
    variation['timelimit'] = 60
    variation['demand'] = True
    variation['std'] = 0.2

    net_dict, ods_dict = dg.read_cases(networks, data_dir)
    dataset, params_opt, params_net, _, _, _ = set_parameters(model, problem, networks[0])

    node_features_base = ods_dict[networks[0]]
    edge_features = net_dict[networks[0]]

    np.random.seed(seed=variation['seed'])

    links = list(edge_features['free_flow'].keys())
    pairs = sorted(list(set([tuple(sorted((i, j))) for (i, j) in links if (j, i) in links])))
    pairs_ls_idx = sorted(np.random.choice(range(len(pairs)), round(variation['swap_pr'] * len(pairs)), replace=False))
    pairs_ls1 = [pairs[i] for i in pairs_ls_idx]
    pairs_ls2 = [(j, i) for (i, j) in pairs_ls1]
    pairs_ls1_idx = [links.index(item) for item in pairs_ls1]
    pairs_ls2_idx = [links.index(item) for item in pairs_ls2]
    test = [[links[pairs_ls1_idx[i]], links[pairs_ls2_idx[i]]] for i in range(len(pairs_ls1_idx))]

    variation['pairs_ls1_idx'] = pairs_ls1_idx
    variation['pairs_ls2_idx'] = pairs_ls2_idx

    if variation['demand']:
        # random perturbation of demand
        node_features = {(a, b): np.amax([0.0, np.random.normal(demand, variation['std'] * demand)])
                         for (a, b), demand in node_features_base.items()}
    else:
        node_features = node_features_base

    return networks[0], model, problem, dataset.test, data_dir, variation, node_features, edge_features, params_net


def prep_gnn(model_name, network, dataset, variation, node_features, edge_features, ga_params, params_net):

    """
    The prep_gnn function prepares the data for the GA fitness evaluation.
        It loads a trained GNN model, and creates a batch of graphs from the population size.
        The function also tiles all relevant parameters to be used in vectorized form by DGL.

    :param model_name: Define the name of the model to be used
    :param network: Define the network to be used
    :param dataset: Load the dataset
    :param variation: Define the number of lanes, max_lanes and budget
    :param edge_features: Store the edge features of a network
    :param ga_params: Pass the parameters of the genetic algorithm to this function
    :param params_net: Pass the parameters of the network to be used
    :return: A list of objects that will be used in the fitness evaluation
    """

    # scenario data
    lanes_n = variation['n_lanes']
    lanes_max = variation['max_lanes']
    pop_size = ga_params['params']['population_size']
    pairs_ls1_idx = variation['pairs_ls1_idx']
    pairs_ls2_idx = variation['pairs_ls2_idx']

    links = list(edge_features['free_flow'].keys())
    fftt = np.array([ff for link, ff in list(edge_features['free_flow'].items())])
    cap_base = np.array([cap for link, cap in list(edge_features['capacity'].items())])
    cap_more = cap_base / lanes_n

    train_data = dataset

    if variation['demand']:
        num_nodes = len(set(itertools.chain.from_iterable(links)))
        demand = np.zeros((num_nodes, num_nodes))
        for (org, dst) in node_features:
            demand[org - 1, dst - 1] = node_features[(org, dst)]
        demand = np.vstack([demand, [0] * len(demand)])

        for graph_idx in range(len(train_data)):
            train_data[graph_idx][0].ndata['feat'] = torch.Tensor(demand).float()

    model_gnn = gnn_model(model_name, params_net)
    # load model
    model_gnn.load_state_dict(torch.load(f'models/model_{network}.pt'))
    model_gnn.eval()
    # ga fitness evaluation data
    ind_data = train_data[0][0]
    pop_data = train_data[:pop_size][0]
    # batch data for pop vectorization
    pop_graphs = dgl.batch(pop_data)
    pop_cap_base = np.tile(cap_base, pop_size)
    pop_cap_more = np.tile(cap_more, pop_size)
    pop_sol_base = np.zeros((pop_size, len(links)))
    pop_init = np.random.choice(a=[0, 1, -1], size=(pop_size, len(pairs_ls1_idx)))
    pop_init[0, :] = np.zeros(len(pairs_ls1_idx))
    pop_init[1, :] = np.ones(len(pairs_ls1_idx))
    pop_init[2, :] = np.ones(len(pairs_ls1_idx)) * -1

    return model_gnn, ind_data, pop_data, cap_base, pop_cap_base, cap_more, pop_cap_more, fftt, links, \
           pop_sol_base, pairs_ls1_idx, pairs_ls2_idx, pop_init


def fitness_batch(model_gnn, data, cap_base, cap_more, pop_sol, ls1_idx, ls2_idx):

    """
    Evaluate population fitness in batches.
    The fitness_batch function is a wrapper for the fitness function. It takes in the same arguments as
    the fitness function, but instead of taking in one solution at a time, it takes in an array of solutions.
    The reason this is necessary is because we want to be able to evaluate multiple solutions at once using
    PyTorch's batching capabilities. The way that PyTorch works with batches requires us to have all data
    for each sample (in our case, each graph) together and contiguous in memory.

    :param model_gnn: Pass the model to the fitness function
    :param data: Pass the data to the fitness function
    :param cap_base: Set the base capacity of each edge
    :param cap_more: Variable capacity of each edge
    :param pop_sol: Store the solutions of the previous generation
    :param ls1_idx: Identify the index of the first link in link pairs (bidirectional links)
    :param ls2_idx: Identify the index of the second link in link pairs (bidirectional links)
    :return: A function that takes a batch of solutions as input and returns the fitness values
    """

    def fitness_function(solutions):

        device = torch.device("cpu")
        pop_size = len(solutions[:, 0])
        sol_size = len(pop_sol[0, :])
        all_size = pop_size * sol_size
        dvs_temp = pop_sol[:pop_size, :]
        dvs_temp[:, ls1_idx] = solutions
        dvs_temp[:, ls2_idx] = solutions * -1
        sol_batch = dvs_temp.flatten()
        graphs = dgl.batch(data[:pop_size]).to(device)
        graph_x = graphs.ndata['feat'].to(device)
        graph_e = graphs.edata['feat'].to(device)
        capacity = cap_base[:all_size] + cap_more[:all_size] * sol_batch
        graph_e[:, 0] = torch.tensor(capacity, dtype=torch.float)

        batch_scores = model_gnn.forward(graphs, graph_x, graph_e)

        ltt_score = graph_e[:, 1] * (1 + 0.15 * torch.pow(torch.div(torch.flatten(batch_scores), graph_e[:, 0]), 4))
        ttt = np.fromiter((ltt_score[b * sol_size: (b + 1) * sol_size] @ batch_scores[b * sol_size: (b + 1) * sol_size]
                           for b in range(pop_size)), 'float')

        return ttt

    return fitness_function


def fitness_graph(model_gnn, data, cap_base, cap_more, sol_base, idx1, idx2):

    """
    Evaluate individual fitness of a solution.
    The fitness_graph returns the fitness of the solution for a given decision variable.

        Parameters:
    :param model_gnn: Pass the model to the fitness function
    :param data: Pass the graph data to the fitness function
    :param cap_base: Set the base capacity of all edges
    :param cap_more: Variable capacity of the edges
    :param sol_base: Store the solution
    :param idx1: Identify the index of the first link in link pairs (bidirectional links)
    :param idx2: Identify the index of the second link in link pairs (bidirectional links)
    :return: A fitness function that takes a solution as an input and returns the total travel time
    """

    def fitness_function(solution):
        device = torch.device("cpu")
        sol_base[idx1] = solution
        sol_base[idx2] = solution * -1
        dvs = sol_base.flatten()
        graphs = data.to(device)
        graph_x = graphs.ndata['feat'].to(device)
        graph_e = graphs.edata['feat'].to(device)
        capacity = cap_base + cap_more * dvs
        graph_e[:, 0] = torch.tensor(capacity, dtype=torch.float)

        batch_scores = model_gnn.forward(graphs, graph_x, graph_e)

        ltt_score = graph_e[:, 1] * (1 + 0.15 * torch.pow(torch.div(torch.flatten(batch_scores), graph_e[:, 0]), 4))
        ttt = np.fromiter(ltt_score @ batch_scores, 'float').sum()

        return ttt

    return fitness_function


def algorithm_GAGNN(model_name, network, dataset, variation, edge_features, node_features, timelimit, params_net):

    """
    The algorithm_GAGNN function is the main function for running the GAGNN (GAGIN) method introduced in:
    (Madadi et al. 2023) https://arxiv.org/abs/2303.06024

    GA implementation using geneticalgorithm2 package:
    https://pypi.org/project/geneticalgorithm2/#report-checker.

    :param model_name: Identify the gnn model
    :param network: Define the network
    :param dataset: Determine the data to be used for training and testing
    :param variation: Problem parameters
    :param edge_features: Pass the edge features to the gnn model
    :param node_features: Define the nodes in the network
    :param timelimit: Set the time limit for running the ga
    :param params_net: Set the parameters of the gnn model
    """

    # set ga parameters
    ga_params = parameters_gagnn()
    # get data to prepare gnn for inference
    model_gnn, ind_data, pop_data, cap_base, pop_cap_base, cap_more, pop_cap_more, fftt, links, pop_sol, idx1, idx2, \
    pop_init = prep_gnn(model_name, network, dataset, variation, node_features, edge_features, ga_params, params_net)

    # initiate ga model
    model_ga = ga(
        function=fitness_graph(model_gnn, ind_data, cap_base, cap_more, pop_sol[0, :], idx1, idx2),
        dimension=len(idx1),
        variable_type='int',
        variable_boundaries=[(-1, 1) for i in range(len(idx1))],
        algorithm_parameters=ga_params['params'])

    t0 = time.time()
    # run ga model
    if ga_params['batch']:
        model_ga.run(
            seed=variation['seed_ga'],
            time_limit_secs=timelimit,
            start_generation=pop_init,
            studEA=ga_params['stud'],
            no_plot=ga_params['no_plot'],
            progress_bar_stream=ga_params['prg_bar'],
            revolution_after_stagnation_step=ga_params['revl'],
            remove_duplicates_generation_step=ga_params['dup_rmv'],
            set_function=fitness_batch(
                model_gnn, pop_data, pop_cap_base, pop_cap_more, pop_sol, idx1, idx2))
    else:
        model_ga.run(
            seed=variation['seed_ga'],
            time_limit_secs=timelimit,
            start_generation=pop_init,
            studEA=ga_params['stud'],
            no_plot=ga_params['no_plot'],
            progress_bar_stream=ga_params['prg_bar'],
            revolution_after_stagnation_step=ga_params['revl'],
            remove_duplicates_generation_step=ga_params['dup_rmv'],
        )

    gagnn_time = min(time.time() - t0, timelimit)

    # process ga output
    output = model_ga.output_dict
    results = model_ga.result
    best_of = model_ga.best_function
    generations = pd.DataFrame(model_ga.report, columns=['Best objective (total travel time)'])
    generations.reset_index(inplace=True)

    generations = generations.rename(columns={'index': 'Generation'})
    _, last_uidx = np.unique(results.last_generation.scores, return_index=True)
    last_ofs = results.last_generation.scores[np.sort(last_uidx)]
    last_dvs = results.last_generation.variables[np.sort(last_uidx, axis=0)]
    top_size = min(ga_params['top'], len(last_ofs))
    top_ofs = last_ofs[:top_size]
    top_dvs = last_dvs[:top_size, :]
    top_fls = np.zeros((top_size, len(links)))

    # check top ranked solutions
    top_ttts = np.ones(top_size) * np.inf
    edge_features_i = copy.deepcopy(edge_features)
    for i in range(top_size):
        dvs_all_links = np.zeros(len(links))
        dvs_all_links[idx1] = top_dvs[i, :]
        dvs_all_links[idx2] = top_dvs[i, :] * -1
        cap = cap_base + cap_more * dvs_all_links
        edge_features_i['capacity'] = dict(zip(links, cap))

        try:
            flows, top_ttts[i], _, _ = dg.due_aeq(node_features, edge_features_i)
            top_fls[i, :] = list(flows.values())
        except:
            pass

    # organize top ranked solutions based on DUE assignment results
    top_results = pd.DataFrame()
    top_results['TTT'] = top_ttts
    top_results['DVs'] = list(top_dvs)
    top_results['Flows'] = list(top_fls)
    top_results = top_results.sort_values(['TTT'], ascending=[True])

    # the best
    best_ttt = top_results.loc[0]['TTT']
    best_dvs = top_results.loc[0]['DVs']
    best_fls = top_results.loc[0]['Flows']

    best_idx_add = [idx1[i] for i in range(len(idx1)) if best_dvs[i] == 1]
    best_idx_rmv = [idx1[i] for i in range(len(idx1)) if best_dvs[i] == -1]

    if any(best_idx_add):
        best_add = np.array(links)[np.array(best_idx_add)]
        print(f'\nAdded lane(s) to:\n {best_add}')
    if any(best_idx_rmv):
        best_rmv = np.array(links)[np.array(best_idx_rmv)]
        print(f'Removed lane(s) from:\n {best_rmv}')

    print('\n********************************************************')
    print('Best OF (with GNN-estimated TTT):\t%.4f' % best_of)
    print('Best TTT (with DUE-calculated TTT):\t%.4f' % best_ttt)
    print('GA-GNN used time:\t%.2f' % gagnn_time)
    print('********************************************************\n')

    return best_ttt, gagnn_time, best_dvs, best_fls, best_of, top_results, generations


if __name__ == "__main__":

    """
    Main function to run the algorithm on a given case individually (mainly for testing).
    """

    case, model, problem, dataset, data_dir, variation, node_features, edge_features, params_net = params_for_test()

    best_ttt, gagnn_time, best_dvs, best_fls, best_of, top_results, generations = algorithm_GAGNN(
        model, case, dataset, variation, edge_features, node_features, variation['timelimit'], params_net)


