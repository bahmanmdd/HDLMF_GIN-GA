"""
   Created by: Bahman Madadi
   Description: generate an instance of D-NDP with lane swaps and solve using SORB algorithm (Wang et al. 2013)
"""

import os
import sys
import time
import copy
import numpy as np
import docplex.mp
import gurobipy
import idaes
from docplex.mp.model import Model
import pyomo.environ as pyo
from pyomo.environ import SolverFactory
from pyomo.opt import SolverStatus, TerminationCondition
import numpy as np
import pandas as pd
import warnings
import data_due_generate as dg
warnings.filterwarnings('ignore')


def parameters_sorb():

    """
    The parameters_sorb function is used to set the parameters for the SORB algorithm.
    :return: A dictionary of parameters for the sorb algorithm
    """

    sorb_params = {}
    sorb_params['mip_solver'] = 'cplex_persistent'  # choices: {'gurobi_persistent', 'cplex_persistent', 'glpk'}
    sorb_params['due_solver'] = 'aeq'  # choices: {'aeq', 'ipp'}
    sorb_params['segments'] = 300
    sorb_params['mip_gap'] = 1e-6
    sorb_params['scale'] = 1e3

    return sorb_params


def params_for_test():

    """
    set parameters for the test function to run the algorithm on a given case individually (mainly for testing)
    :return: A dictionary of parameters for the sorb algorithm
    """

    case = ['Anaheim']
    case = ['SiouxFalls']
    case = ['Eastern-Massachusetts']
    data_dir = 'TransportationNetworks'
    variation = {}
    variation['seed'] = 0
    variation['problem'] = 'NDP_LS'
    variation['swap_pr'] = 0.25
    variation['n_lanes'] = 2  # assumed number of existing lanes per link
    variation['max_lanes'] = 1  # max number of new lanes to add to each link
    variation['timelimit'] = 300
    net_dict, ods_dict = dg.read_cases(case, data_dir)
    node_features = ods_dict[case[0]]
    edge_features = net_dict[case[0]]

    np.random.seed(seed=variation['seed'])

    links = list(edge_features['free_flow'].keys())
    pairs = sorted(list(set([tuple(sorted(t)) for t in links])))
    pairs_ls_idx = np.random.choice(range(len(pairs)), round(variation['swap_pr']*len(pairs)), replace=False)
    pairs_ls1 = list(set([pairs[i] for i in pairs_ls_idx]))
    pairs_ls2 = [(j, i) for (i, j) in pairs_ls1]
    pairs_ls1_idx = [i for i in range(len(links)) if links[i] in pairs_ls1]
    pairs_ls2_idx = [i for i in range(len(links)) if links[i] in pairs_ls2]
    links_ls = sorted(pairs_ls1 + pairs_ls2)
    links_ls_idx = [i for i in range(len(links)) if links[i] in links_ls]

    variation['links_ls'] = links_ls
    variation['pairs_ls'] = pairs_ls1
    variation['links_ls_idx'] = links_ls_idx
    variation['pairs_ls_idx'] = pairs_ls_idx
    variation['pairs_ls1_idx'] = pairs_ls1_idx
    variation['pairs_ls2_idx'] = pairs_ls2_idx

    return case[0], data_dir, variation, node_features, edge_features


# SO-relaxation-based algorithm (Wang et al. 2013): https://www.sciencedirect.com/science/article/abs/pii/S0191261513000179
# an adaptation of the implementation of DNDP with link addition by David Ray: https://github.com/davidrey123/DNDP
def algorithm_SORB(node_features, edge_features, variation):

    """
    The algorithm_SORB function is the main function for the SORB algorithm.
    It takes in node_features, edge_features, and variation as inputs.
    The node_features and edge_features are pandas dataframes that contain information about each of the nodes and edges in a network respectively.
    The variation dictionary contains information about how to run this algorithm (e.g., timelimit).
    This function returns result, ub_sorb, gap_sorb, time_sorb, best dvs upper values (best DVs from RP), best dvs lower values (best DVs from UE)

    :param node_features: Node features of the network
    :param edge_features: Edge features of the graph
    :param variation: Set the time limit for the algorithm
    """

    params = parameters_sorb()
    print('Modeling the relaxed problem for system optimal relaxation based (SORB) algorithm...')

    time_limit = variation['timelimit']
    t0 = time.time()
    best_dvs_upper = {}
    best_dvs_lower = {}
    result_list = []
    dvs_rp_all = []
    result = []
    gap_sorb = np.inf
    ub_sorb = np.inf
    n_it = 0
    converged = False

    while not converged:

        # check remaining time budget
        t_remain = time_limit - (time.time() - t0)
        if t_remain <= 0:
            print('time limit exceeded, terminating...')
            break

        # model and solve the RP
        rp_dvs, rp_ttt, rp_cap, rp_time = rp_model_pwl_swp1_solve(
            node_features, edge_features, variation, params, dvs_rp_all, t_remain)
        dvs_rp_all.append(rp_dvs)

        # check feasibility
        if rp_ttt == -1:
            if n_it == 0:
                print('Relaxed problem is infeasible! Terminating...')
            else:
                print('No feasible interdiction cut found! Terminating...')
            break

        # solve UE with y_opt (updated lane capacities) and get total travel time
        edge_features_temp = copy.deepcopy(edge_features)
        edge_features_temp['capacity'] = rp_cap

        if params['due_solver'] == 'aeq':
            ue_dvs, ue_ttt, ue_time, _, = dg.due_aeq(node_features, edge_features_temp)
        else:
            ue_dvs, ue_ttt, ue_time, _, = dg.due_ipp(node_features, edge_features_temp)

        if n_it == 0:
            print('*************************************************************')
            print('System optimal relaxation based (SORB) algorithm initiated...')
            print('*************************************************************')
            print('*******************************************************************************')
            print('-------------------------------------------------------------------------------')
            print(
                '\n%15s\t%15s\t%15s\t%15s\t%15s' % ('Iteration', 'RP TTT', 'UE TTT', 'RP Time', 'UE Time'))
            print('-------------------------------------------------------------------------------')

        print('%15d\t%15.4f\t%15.4f\t%15.4f\t%15.4f' % (n_it, rp_ttt, ue_ttt, rp_time, ue_time))
        result_list.append([n_it, rp_ttt, ue_ttt, rp_time, ue_time])

        # update UB if new UE solution better than bound
        if ue_ttt < ub_sorb:
            print('update UB')
            ub_sorb = ue_ttt
            best_dvs_upper = rp_dvs
            best_dvs_lower = ue_dvs

        # calculate gap
        if rp_ttt >= ub_sorb:
            gap_sorb = 0.0
        else:
            gap_sorb = (ub_sorb - rp_ttt) / rp_ttt

        # check convergence
        if rp_ttt >= ub_sorb:
            converged = True

        n_it += 1

    time_sorb = min((time.time() - t0), time_limit)
    if result_list:
        result = pd.DataFrame(np.array(result_list), columns=['Iteration', 'RP TTT', 'UE TTT', 'RP Time', 'UE Time'])

    bes_lanes = {a: b for a, b in best_dvs_upper.items() if b != 0}
    print('\nBest DVs:')
    print(bes_lanes)

    print('***********************************')
    print('GAP:\t%.4f' % gap_sorb)
    print('Best Upper Bound:\t%.4f' % ub_sorb)
    print('Total Time for SORB:\t%.2f' % time_sorb)
    print('***********************************')

    return result, ub_sorb, gap_sorb, time_sorb, best_dvs_upper.values(), best_dvs_lower.values()


# piecewise linear with 1 swap lane
def rp_model_pwl_swp1_solve(node_features, edge_features, variation, params, dvs_rp_all, t_remain):

    """
    The following function solves the problem with SORB using a piecewise linear approximation of the link
    travel time function with 1 swap lane per link.

    :param node_features: Store the demand between each pair of nodes
    :param edge_features: Store the edge features of the graph
    :param variation: Determine the number of lanes and budget
    :param params: Set the parameters for the rp_model_pwl2_solve function
    :param dvs_rp_all: Store the previous solutions
    :param t_remain: Set a time limit on the rp_model_pwl2_solve function
    :return: The following:
        rp_dvs: The solution of the rp_model_pwl2_solve function
        rp_ttt: The total travel time of the solution
        rp_cap: The solution of the rp_model_pwl2_solve function
        rp_time: The time required to solve the rp_model_pwl2_solve function
    """

    t0_rp = time.time()
    fftt = edge_features['free_flow']
    beta = edge_features['beta']
    alpha = edge_features['alpha']
    links = list(fftt.keys())
    scale = params['scale']
    n_segs = params['segments']
    lanes_n = variation['n_lanes']

    links_ls = variation['links_ls']

    nodes = np.unique([list(edge) for edge in links])
    capacity = {(a, b): cap/scale for (a, b), cap in edge_features['capacity'].items()}
    # capacity_var = {(i, j): capacity[i, j]/lanes_n for (i, j) in links}
    capacity_var = {(i, j): (capacity[i, j]/lanes_n if (i, j) in links_ls else 0) for (i, j) in links}

    origins = np.unique([a_node for (a_node, b_node) in list(node_features.keys())])
    destinations = np.unique([int(b_node) for (a_node, b_node) in list(node_features.keys())])

    coef_0 = {(i, j): (fftt[i, j] * alpha[i, j])/(capacity[i, j] ** beta[i, j]) for (i, j) in links}
    coef_1 = {(i, j): (fftt[i, j] * alpha[i, j])/((capacity[i, j]+capacity_var[i, j]) ** beta[i, j]) for (i, j) in links}
    coef_2 = {(i, j): (fftt[i, j] * alpha[i, j])/((capacity[i, j]-capacity_var[i, j]) ** beta[i, j]) for (i, j) in links}

    # link travel time function linear approximation using m uniform segments
    # maximum link flow is instance-specific: value is calibrated for Sioux Falls
    m_flow = 1e5 / scale
    V = set([i for i in range(0, n_segs + 1)])
    Vp = V.difference({0})
    a = {(i, j, v): float() for (i, j) in links for v in V}
    for (i, j) in links:
        cnt = 0
        step = m_flow / (len(V) - 1)
        for v in V:
            a[i, j, v] = cnt * step
            cnt += 1

    # create node-destination demand matrix (not a regular OD!)
    demand_total = 0
    demand = {(n, d): 0 for n in nodes for d in destinations}
    for r in origins:
        for s in destinations:
            demand[r, s] = node_features[r, s] / scale
            demand_total += demand[r, s]
    for s in destinations:
        demand[s, s] = - sum(demand[j, s] for j in origins)

    ########################################
    # initiate model (this is a cplex model)
    model_rp = Model(name='SORB', log_output=False)

    # decision variables
    dvs_upper = {(i, j, d): model_rp.binary_var() for (i, j) in links for d in [1, 2]}
    dvs_lower = {(i, j, s): model_rp.continuous_var() for (i, j) in links for s in destinations}
    pwll = {(i, j, v): model_rp.continuous_var() for (i, j) in links for v in V}
    pwlr = {(i, j, v): model_rp.continuous_var() for (i, j) in links for v in V}

    # swap constraint
    for (i, j) in links:
        if (i, j) in links_ls:
            model_rp.add_constraint(dvs_upper[i, j, 1] == dvs_upper[j, i, 2])
            model_rp.add_constraint(dvs_upper[j, i, 1] == dvs_upper[i, j, 2])
            model_rp.add_constraint(sum(dvs_upper[i, j, d] for d in [1, 2]) <= 1)
        else:
            model_rp.add_constraint(dvs_upper[i, j, 1] == 0)
            model_rp.add_constraint(dvs_upper[i, j, 2] == 0)

    # demand satisfaction constraints
    for i in nodes:
        for s in destinations:
            model_rp.add_constraint(
                sum(dvs_lower[i, j, s] for j in nodes if (i, j) in links) -
                sum(dvs_lower[j, i, s] for j in nodes if (j, i) in links) == demand[i, s])

    # pwl approximations
    for (i, j) in links:
        model_rp.add_constraint(
            sum(dvs_lower[i, j, s] for s in destinations) ==
            sum(pwll[i, j, v] * a[i, j, v - 1] + pwlr[i, j, v] * a[i, j, v] for v in Vp))
        model_rp.add_constraint(
            sum(pwll[i, j, v] + pwlr[i, j, v] for v in V) == 1)

    # non-negativities
    for (i, j) in links:
        for s in destinations:
            model_rp.add_constraint(dvs_lower[i, j, s] >= 0)
        for v in V:
            model_rp.add_constraint(pwll[i, j, v] >= 0)
            model_rp.add_constraint(pwlr[i, j, v] >= 0)

    # interdiction cuts
    if dvs_rp_all:
        for solution in dvs_rp_all:
            model_rp.add_constraint(sum(
                    (1 - solution[i, j, s]) * dvs_upper[i, j, s] +
                    (1 - dvs_upper[i, j, s]) * solution[i, j, s]
                    for (i, j) in links for s in [1, 2]) >= 1)

    model_rp.minimize(sum(fftt[i, j] * sum(dvs_lower[i, j, s] for s in destinations) +
                          (coef_0[i, j] * (1 - dvs_upper[i, j, 1] - dvs_upper[i, j, 2]) +
                           coef_1[i, j] * dvs_upper[i, j, 1] +
                           coef_2[i, j] * dvs_upper[i, j, 2]) *
                          sum(pwll[i, j, v] * (a[i, j, v - 1] ** (beta[i, j] + 1)) +
                              pwlr[i, j, v] * (a[i, j, v] ** (beta[i, j] + 1)) for v in Vp)
                          for (i, j) in links))

    model_rp.parameters.threads = 1
    model_rp.parameters.mip.display = 0
    model_rp.parameters.timelimit = t_remain
    model_rp.parameters.mip.tolerances.mipgap = params['mip_gap']
    ########################################

    rp_ttt = -1
    rp_dvs = {}
    rp_cap = {}
    sol = model_rp.solve()
    try:
        rp_ttt = model_rp.objective_value * scale
        rp_dvs = {(i, j, d): round(model_rp.solution.get_value(dvs_upper[i, j, d])) for (i, j) in links for d in [1, 2]}
        rp_cap = {(i, j): (capacity[i, j] + rp_dvs[i, j, 1] * capacity_var[i, j] - rp_dvs[i, j, 2] * capacity_var[i, j])
                          * scale for (i, j) in links}

        rp_lf = {}
        for (i, j) in links:
            rp_lf[i, j] = sum(model_rp.solution.get_value(dvs_lower[i, j, s]) for s in destinations) * scale
            # rp_ttt += rp_lf[i, j] * (fftt[i, j] * (1 + alpha[i, j] * ((rp_lf[i, j]/rp_cap[i, j]) ** beta[i, j])))
    except Exception:
        print('status:\t%s' % model_rp.solve_details.status)

    rp_time = time.time() - t0_rp

    return rp_dvs, rp_ttt, rp_cap, rp_time


if __name__ == "__main__":

    """
    Main function to run the algorithm on a given case individually (mainly for testing).
    """

    case, data_dir, variation, node_features, edge_features = params_for_test()

    print('--------')
    print(f'Network: {case}')
    print('--------')

    results, ub, gap, ct, best_dvs_upper, best_dvs_lower = algorithm_SORB(node_features, edge_features, variation)
    if best_dvs_upper:
        nonzeros = sum(best_dvs_upper)

    print(f'{nonzeros}')

    test = True


