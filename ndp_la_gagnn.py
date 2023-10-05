"""
   Created by: Bahman Madadi
   Description: generate an instance of D-NDP with lane additions and solve using GA-GNN algorithm (Madadi 2023):
   https://arxiv.org/abs/2303.06024
"""

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
            mutation_probability - probability of mutation for each gene in a chromosome;  # 0.05, 0.01, 0.005, ... , 1e-6?
            elit_ratio - ratio between elite and non-elite chromosomes in a new generation (0 &lt; elit &lt; 1);  # 0.3
            parents_portion - ratio between parents and offspring in a new generation (0 &lt; parents_portion &lt; 1);  # 0.4
            crossover_type - 'one_point', 'two_point', 'uniform', 'segment', 'shuffle'
            mutation_type - 'uniform_by_center'
            selection_type - 'roulette', 'stochastic', 'sigma_scaling', 'ranking', 'linear_ranking', 'tournament'
            max_iteration_without_improv - maximum number of iterations without improvement

    :return: A dictionary of parameters for the ga-gnn algorithm
    """

    ga_gnn = {}
    ga_gnn['params'] = {'max_num_iteration': 10000000,
                        'population_size': 128,
                        'mutation_probability': 0.05,
                        'elit_ratio': 0.3,
                        'parents_portion': 0.4,
                        'crossover_type': 'uniform',  # 'one_point', 'two_point', 'uniform', 'segment', 'shuffle'
                        'mutation_type': 'uniform_by_center',
                        'selection_type': 'ranking',
                        'max_iteration_without_improv': 10000}
    ga_gnn['penalty'] = 500
    ga_gnn['batch'] = True
    ga_gnn['top'] = 20
    ga_gnn['stud'] = False
    ga_gnn['revl'] = None  # start revolution after x generations of stagnation
    ga_gnn['dup_rmv'] = 1  # remove duplicates after x generations

    ga_gnn['no_plot'] = False
    ga_gnn['prg_bar'] = None

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
    variation['problem'] = 'NDP_LA'
    variation['budget'] = 0.5
    variation['n_lanes'] = 2  # assumed number of existing lanes per link
    variation['max_lanes'] = 1  # max number of new lanes to add to each link
    variation['timelimit'] = 60
    net_dict, ods_dict = dg.read_cases(networks, data_dir)
    dataset, params_opt, params_net, _, _, _ = set_parameters(model, problem, networks[0])

    return networks[0], model, problem, dataset.test, data_dir, variation, ods_dict[networks[0]], net_dict[networks[0]], params_net


def prep_gnn(model_name, network, dataset, variation, edge_features, ga_params, params_net):

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
    budget_portion = variation['budget']
    pop_size = ga_params['params']['population_size']

    fftt = np.array([ff for link, ff in list(edge_features['free_flow'].items())])
    links = [link for link, cap in list(edge_features['capacity'].items())]
    cap_base = np.array([cap for link, cap in list(edge_features['capacity'].items())])
    cap_more = cap_base / lanes_n
    lane_costs = fftt * cap_base
    budget = budget_portion * np.sum(lane_costs)

    model_gnn = gnn_model(model_name, params_net)
    train_data = dataset
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
    pop_lane_cost = np.tile(lane_costs, pop_size)

    pop_sol_base = np.zeros((pop_size, len(links)))
    pop_init = np.random.choice(a=[0, 1], size=(pop_size, len(links)), p=[1 - budget_portion, budget_portion])
    pop_init[0, :] = np.zeros(len(links))
    pop_init[1, :] = np.ones(len(links))

    return model_gnn, ind_data, pop_data, cap_base, pop_cap_base, cap_more, pop_cap_more, lane_costs, pop_lane_cost, \
           budget, fftt, links, pop_init


def fitness_population(model_gnn, data, cap_base, cap_more, cost, budget, penalty):

    """
    Evaluate population fitness in batches.
    The fitness_population function takes in the following parameters:
        model_gnn (torch.nn.Module): The GNN model to be used for fitness evaluation
        data (list of dgl graphs): A list of DGL graphs representing the population to be evaluated
        cap_base (np.array): An array containing the base capacities for each edge in each graph
            in data, flattened into a single array with shape [num_edges * num_graphs]
            where num_edges is equal across all graphs and num_graphs is len(data)

    :param model_gnn: Pass the trained gnn model to the fitness function
    :param data: Pass the graph data to the fitness function
    :param cap_base: Set the base capacity of each edge in the graph
    :param cap_more: Variable capacity of each edge in the graph
    :param cost: Calculate the cost of each solution
    :param budget: Calculate the violation
    :param penalty: Penalize solutions that exceed the budget
    """

    def fitness_function(solutions):
        sol_batch = solutions.flatten()
        device = torch.device("cpu")
        pop_size = len(solutions[:, 0])
        sol_size = len(solutions[0, :])
        all_size = pop_size * sol_size
        graphs = dgl.batch(data[:pop_size]).to(device)
        # graphs = data.to(device)
        graph_x = graphs.ndata['feat'].to(device)
        graph_e = graphs.edata['feat'].to(device)
        capacity = cap_base[:all_size] + cap_more[:all_size] * sol_batch
        graph_e[:, 0] = torch.tensor(capacity, dtype=torch.float)

        batch_scores = model_gnn.forward(graphs, graph_x, graph_e)

        ltt_score = graph_e[:, 1] * (1 + 0.15 * torch.pow(torch.div(torch.flatten(batch_scores), graph_e[:, 0]), 4))
        cost_score = cost[:all_size] * sol_batch
        ttt = np.fromiter((ltt_score[b * sol_size: (b + 1) * sol_size] @ batch_scores[b * sol_size: (b + 1) * sol_size]
                           for b in range(pop_size)), 'float')

        violation = penalty * np.array([max(0, np.sum(cost_score[b * sol_size: (b + 1) * sol_size]) - budget)
                                        for b in range(pop_size)])
        fitness = ttt + violation

        return fitness

    return fitness_function


def fitness_graph(model_gnn, data, cap_base, cap_more, cost, budget, penalty):

    """
    Evaluate individual fitness of a solution.
    The fitness_graph function takes in a model_gnn, data, cap_base, cap_more, cost and budget.
    It returns a fitness function that can be used to evaluate the performance of an individual solution.

    :param model_gnn: Pass the trained gnn model to the fitness function
    :param data: Pass the graph data to the fitness function
    :param cap_base: Set the base capacity of all edges
    :param cap_more: Variable capacity of each edge
    :param cost: Calculate the cost of a solution
    :param budget: Set the maximum cost of the solution
    :param penalty: Penalize the solution if it exceeds the budget
    :return: A function that takes a solution as input and returns the fitness of that solution
    """

    def fitness_function(solution):
        device = torch.device("cpu")
        graphs = data.to(device)
        graph_x = graphs.ndata['feat'].to(device)
        graph_e = graphs.edata['feat'].to(device)
        capacity = cap_base + cap_more * solution
        graph_e[:, 0] = torch.tensor(capacity, dtype=torch.float)

        batch_scores = model_gnn.forward(graphs, graph_x, graph_e)

        ltt_score = graph_e[:, 1] * (1 + 0.15 * torch.pow(torch.div(torch.flatten(batch_scores), graph_e[:, 0]), 4))
        ttt = np.fromiter(ltt_score @ batch_scores, 'float').sum()

        violation = penalty * max(0, (cost @ solution) - budget)
        fitness = ttt + violation

        return fitness

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
    model_gnn, ind_data, pop_data, cap_base, pop_cap_base, cap_more, pop_cap_more, lane_costs, pop_lane_cost, budget, \
    fftt, links, pop_init = prep_gnn(model_name, network, dataset, variation, edge_features, ga_params, params_net)

    # initiate ga model
    model_ga = ga(
        function=fitness_graph(model_gnn, ind_data, cap_base, cap_more, lane_costs, budget, ga_params['penalty']),
        dimension=len(cap_base),
        variable_type='bool',
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
            set_function=fitness_population(
                model_gnn, pop_data, pop_cap_base, pop_cap_more, pop_lane_cost, budget, ga_params['penalty']))
    else:
        model_ga.run(seed=variation['seed_ga'],
                     time_limit_secs=timelimit,
                     start_generation=pop_init,
                     studEA=ga_params['stud'],
                     no_plot=ga_params['no_plot'],
                     progress_bar_stream=ga_params['prg_bar'],
                     revolution_after_stagnation_step=ga_params['revl'],
                     remove_duplicates_generation_step=ga_params['dup_rmv'])

    gagnn_time = min(time.time() - t0, timelimit)

    # process ga output
    output = model_ga.output_dict
    results = model_ga.result
    best_of = model_ga.best_function
    generations = pd.DataFrame(model_ga.report, columns=['Best objective (TTT + Penalty)'])
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
    top_feas = np.zeros(top_size)
    edge_features_i = copy.deepcopy(edge_features)
    for i in range(top_size):
        if (lane_costs @ top_dvs[i, :]) <= budget:
            top_feas[i] = 1
            cap = cap_base + cap_more * top_dvs[i, :]
            edge_features_i['capacity'] = dict(zip(links, cap))
            try:
                flows, top_ttts[i], _, _ = dg.due_aeq(node_features, edge_features_i)
                top_fls[i, :] = list(flows.values())
            except:
                pass

    # organize top ranked solutions based on DUE assignment results
    top_results = pd.DataFrame()
    top_results['TTT'] = top_ttts
    top_results['Feasible'] = top_feas
    top_results['DVs'] = list(top_dvs)
    top_results['Flows'] = list(top_fls)
    top_results = top_results.sort_values(['Feasible', 'TTT'], ascending=[False, True])

    # the best
    best_ttt = top_results.loc[0]['TTT']
    best_fsb = bool(top_results.loc[0]['Feasible'])
    best_dvs = top_results.loc[0]['DVs']
    best_fls = top_results.loc[0]['Flows']

    print('\n********************************************************')
    print('Best OF (with GNN estimated TTT):\t%.4f' % best_of)
    print('Best TTT (with DUE-calculated TTT):\t%.4f' % best_ttt)
    print(f'Solution feasible: {best_fsb}')
    print('GA-GNN used time:\t%.2f' % gagnn_time)
    print('********************************************************\n')

    return best_ttt, best_fsb, gagnn_time, best_dvs, best_fls, best_of, top_results, generations


if __name__ == "__main__":

    """
    Main function for testing GAGNN
    """

    case, model, problem, dataset, data_dir, variation, node_features, edge_features, params_net = params_for_test()

    best_ttt, best_fsb, gagnn_time, best_dvs, best_fls, best_of, top_results, generations = algorithm_GAGNN(
        model, case, dataset, variation, edge_features, node_features, variation['timelimit'], params_net)




