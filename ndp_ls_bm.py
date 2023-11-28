"""
   Created by: Bahman Madadi
   Description: Description: initiate (generate) multiple instances of D-NDP with lane swap and solve using:
                    1. SORB algorithm (Wang et al. 2013)
                    2. GA with a (trained) GNN estimator as fitness function: https://arxiv.org/abs/2303.06024
"""

import glob
import os
import time
import numpy as np
import pandas as pd
import scipy.stats as st
import matplotlib.pyplot as plt

import data_due_generate as dg
from train_test import set_parameters
from ndp_ls_sorb import algorithm_SORB, parameters_sorb
from ndp_ls_gagnn import algorithm_GAGNN, parameters_gagnn


def scenarios():

    """
    The scenarios function is used to define the scenarios that will be run.
    """

    variation = {}
    variation['seed'] = 10
    variation['reps'] = 5
    variation['seed_ga'] = 10
    variation['problem'] = 'NDP_LS'
    variation['model_gnn'] = 'GIN'
    variation['problem_ll'] = 'DUE'
    variation['run_bm'] = False

    variation['demand'] = True
    variation['std'] = 0.2

    networks = ['SiouxFalls', 'Eastern-Massachusetts', 'Anaheim']
    nets_dir = 'TransportationNetworks'
    methods = ['SORB', 'GAGNN']  # the first one should be an exact solution method

    max_time = 60 * 60 * 4
    list_time = [60]
    list_swpr = [0.25, 0.5, 0.75]
    list_lanes_now = [2, 4]
    list_lanes_add = [1]

    return variation, networks, nets_dir, list_time, list_swpr, list_lanes_now, list_lanes_add, methods, max_time


def generate_ndp_ls(variation, edge_features, node_features_base):

    """
    The generate_ndp_ls function generates a new network design problem (NDP) with link-swapping (LS) variation.

    :param variation: Generate the variation based on perturbation of the network and demand
    :param edge_features: Generate the links_ls list
    :param node_features_base: Generate the node_features dictionary
    :return: A variation dictionary and a node_features dictionary
    """

    np.random.seed(seed=variation['seed'])

    links = list(edge_features['free_flow'].keys())
    pairs = sorted(list(set([tuple(sorted((i, j))) for (i, j) in links if (j, i) in links])))
    pairs_ls_idx = sorted(np.random.choice(range(len(pairs)), round(variation['swap_pr'] * len(pairs)), replace=False))
    pairs_ls1 = [pairs[i] for i in pairs_ls_idx]
    pairs_ls2 = [(j, i) for (i, j) in pairs_ls1]
    pairs_ls1_idx = [links.index(item) for item in pairs_ls1]
    pairs_ls2_idx = [links.index(item) for item in pairs_ls2]
    links_ls = sorted(pairs_ls1 + pairs_ls2)
    variation['links_ls'] = links_ls
    variation['pairs_ls1_idx'] = pairs_ls1_idx
    variation['pairs_ls2_idx'] = pairs_ls2_idx

    if variation['demand']:
        # random perturbation of demand
        node_features = {(a, b): np.amax([0.0, np.random.normal(demand, variation['std'] * demand)])
                         for (a, b), demand in node_features_base.items()}
    else:
        node_features = node_features_base

    return variation, node_features


def save_params(params, out_dir, method):

    """
    The save_params function saves the parameters of a given scenario to a text file.
    """

    with open(out_dir + '/params_' + method + '.txt', 'w') as f:
        f.write("""Algorithm: {}\n\nParameters={}\n\n\n""" .format(method, params))


def save_best_dvs(name, out_dir, best_dvs, ct, budget, lanes_now, lanes_add):

    """
    The save_best_dvs function saves the best dvs of a given scenario to a csv.
    """

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    if any(best_dvs):
        pd.DataFrame(best_dvs).to_csv(f'{out_dir}/{name}_ct-{ct}_bg-{budget}_l-{lanes_now}_la-{lanes_add}.csv', index=False)


def save_results_scenario(out_dir, results, ct, budget, lanes_now, lanes_add, rep=0):

    """
    The save_results_scenario function saves the results of a given scenario to a csv.
    """

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    if len(results) != 0:
        results.to_csv(f'{out_dir}/Results_ct-{ct}_bg-{budget}_l-{lanes_now}_la-{lanes_add}_R-{rep}.csv', index=False)


def plot_results_scenario(out_dir, results, ct, budget, lanes_now, lanes_add, rep=0):

    """
    The plot_results_scenario function plots the results of a given scenario.
    """

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    if len(results) != 0:
        results.set_index(list(results)[0]).plot()
        plt.savefig(f'{out_dir}/Iterations_ct-{ct}_bg-{budget}_l-{lanes_now}_la-{lanes_add}_R-{rep}.jpeg', dpi=600)
        plt.close("all")


def benchmark_ndp_ls():

    """
    The benchmark_ndp_ls function is the main function of this script. It runs a series of experiments
    to compare the performance of different algorithms for solving an NDP-LS problem. The experiments are run on
    a set of networks, and each experiment consists in finding a solution to an NDP-LS problem with given parameters:
        - percentage (swap_pr) and number (n_lanes) lanes per link; maximum number (max_lanes) added lanes per link;
        - time limit for each algorithm.
    """

    t0 = time.time()
    variation, networks, nets_dir, list_time, list_swpr, list_lanes_now, list_lanes_add, methods, max_time = scenarios()
    net_dict, ods_dict = dg.read_cases(networks, nets_dir)

    for case in networks:
        node_features_base = ods_dict[case]
        edge_features = net_dict[case]
        prob = variation['problem']

        names_reps = ['Method', 'CT_max', 'CT_used', 'SW_RP', 'Cap', 'TTT', 'Gap']
        names_summary = ['Method', 'CT_max', 'CT_used', 'SW_PR', 'Cap', 'TTT_min', 'TTT_AVG', 'TTT_max',
                         'Gap_min', 'Gap_AVG', 'Gap_max', 'CI_LB', 'CI_UB']
        summaries = {}
        replications = {}
        for method in methods:
            replications[method] = pd.DataFrame(columns=names_reps)
            summaries[method] = pd.DataFrame(columns=names_summary)

        dataset_full, _, params_net, _, _, _ = set_parameters(variation['model_gnn'], variation['problem_ll'], case)
        dataset = dataset_full.test
        del dataset_full

        for swpr in list_swpr:
            # prep variation
            variation['swap_pr'] = swpr
            variation, node_features = generate_ndp_ls(variation, edge_features, node_features_base)
            for lanes_now in list_lanes_now:
                for lanes_add in list_lanes_add:
                    variation['n_lanes'] = lanes_now
                    variation['max_lanes'] = lanes_add
                    cap = lanes_add/lanes_now

                    # first solve with an exact solver (or read results from file if we have done it before)
                    method = methods[0]
                    variation['timelimit'] = max_time
                    out_dir = f'output/{prob}/{case}/{method}'

                    # read from file
                    if not variation['run_bm']:
                        bm_filename = glob.glob(out_dir + "/reps_*.csv")
                        bm = pd.read_csv(bm_filename[0])
                        ttt_bm = bm[(bm['CT_max'] == max_time) & (bm['SW_RP'] == swpr) & (bm['Cap'] == cap)]['TTT']

                    # solve
                    else:

                        print('\n\n****************************************************************')
                        print('****************************************************************')
                        print(f'{case} network')
                        print('Scenario:')
                        print('---------')
                        print(f'LS percentage: {swpr*100} percent of all links')
                        print(f'Lanes per link: {lanes_now}')
                        print(f'Max added lanes: {lanes_add}')
                        print('************************************************************************************')
                        print(f'Attempting to find an optimal solution using method: {method}')
                        print(f'Time limit: {max_time} seconds')
                        print('************************************************************************************')

                        # solve with SORB
                        results, ttt_bm, gap, ct_method, best_dvs_upper, best_dvs_lower = \
                            algorithm_SORB(node_features, edge_features, variation)

                        variant_out = np.array([method, max_time, ct_method, swpr, cap, ttt_bm, gap])
                        replications[method].loc[len(replications[method])] = variant_out
                        save_results_scenario(out_dir, results, max_time, swpr, lanes_now, lanes_add)
                        if any(results):
                            plot_results_scenario(out_dir, results[['Iteration', 'RP TTT', 'UE TTT']], max_time, swpr, lanes_now, lanes_add)
                            save_best_dvs('DVs_UL', out_dir, best_dvs_upper, max_time, swpr, lanes_now, lanes_add)
                            save_best_dvs('DVs_LL', out_dir, best_dvs_lower, max_time, swpr, lanes_now, lanes_add)
                            params = parameters_sorb()
                            save_params(params, out_dir, method)

                    for ct in list_time:
                        # the rest of the methods
                        for method in methods[1:]:

                            out_dir = f'output/{prob}/{case}/{method}'
                            print('\n\n*******************************************************************************')
                            print('*******************************************************************************')
                            print(f'{case} network')
                            print('Scenario:')
                            print('---------')
                            print(f'Budget: {swpr * 100} percent of total cost')
                            print(f'Lanes per link: {lanes_now}')
                            print(f'Max added lanes: {lanes_add}')
                            print('***********************************************************************************')
                            print(f'Method: {method}')
                            print(f'Time limit: {ct} seconds')
                            print('***********************************************************************************')

                            ttts = np.zeros((variation['reps']))
                            gaps = np.zeros((variation['reps']))

                            for r in range(variation['reps']):
                                print(f'Replication {r+1}')
                                print('-------------')
                                variation['seed_ga'] = r

                                # solve with GA-GNN
                                if method == 'GAGNN':
                                    ttts[r], ct_method, best_dvs, best_fls, best_of, top_results, generations = \
                                        algorithm_GAGNN(variation['model_gnn'], case, dataset, variation, edge_features,
                                                        node_features, ct, params_net)

                                # save replication results
                                gaps[r] = (ttts[r] - ttt_bm) / ttt_bm
                                rep_row = np.array([method, ct, ct_method, swpr, cap, ttts[r], gaps[r]])
                                replications[method].loc[len(replications[method])] = rep_row
                                plot_results_scenario(out_dir, generations, ct, swpr, lanes_now, lanes_add, r)
                                if method == 'GAGNN':
                                    save_results_scenario(out_dir, top_results, ct, swpr, lanes_now, lanes_add, r)

                            # save summary stats of replications
                            (cilb, ciub) = st.t.interval(0.95, len(gaps)-1, loc=np.mean(gaps), scale=st.sem(gaps))
                            sum_row = np.array([method, ct, ct_method, swpr, cap,
                                                np.min(ttts), np.mean(ttts), np.max(ttts),
                                                np.min(gaps), np.mean(gaps), np.max(gaps),
                                                cilb, ciub])
                            summaries[method].loc[len(summaries[method])] = sum_row
                            params = parameters_gagnn()
                            save_params(params, out_dir, method)

        # save overall stats
        now = time.strftime("%m%d-%H%M")
        for method in methods[1:]:
            replications[method].to_csv(f'output/{prob}/{case}/{method}/reps_{now}.csv', index=False)
            replications[method].to_csv(f'output/{prob}/{case}/reps_{method}_{now}.csv', index=False)
            summaries[method].to_csv(f'output/{prob}/{case}/{method}/summary_{now}.csv', index=False)
            summaries[method].to_csv(f'output/{prob}/{case}/summary_{method}_{now}.csv', index=False)
        if variation['run_bm']:
            method = methods[0]
            replications[method].to_csv(f'output/{prob}/{case}/{method}/reps_{now}.csv', index=False)
            replications[method].to_csv(f'output/{prob}/{case}/reps_{method}_{now}.csv', index=False)

        print(f'Total experiment time for {case} network: {float(time.time() - t0)/3600} hours')
        print(f'Done with {case} network!\n')
        print('*************************************')
        print('*************************************\n\n')
    print('Done with all!')
    print('**************\n\n')


if __name__ == "__main__":
    benchmark_ndp_ls()




