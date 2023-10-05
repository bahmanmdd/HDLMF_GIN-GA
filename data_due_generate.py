"""
   Created by: Bahman Madadi
   Description: generate a dataset with solved instances of DUE for each network in benchmark networks
"""

import os
import time
import copy
import numpy as np
import pandas as pd
import pyomo.environ as pyo
from pyomo.environ import SolverFactory
from pyomo.opt import SolverStatus, TerminationCondition
import idaes
import pickle
import warnings

from aequilibrae.matrix import AequilibraeMatrix
from aequilibrae.paths import Graph
from aequilibrae.paths import TrafficAssignment
from aequilibrae.paths.traffic_class import TrafficClass
from data_dataset_prep import DUEDatasetDGL
warnings.filterwarnings('ignore')


# define parameters for datasets
def parameters():
    """
       parameters for generating dataset(s) with solved instances of DUE
    """

    dataset = {}
    dataset['n_samples'] = 10000

    input = {}
    # Available networks: ['SiouxFalls', 'Eastern-Massachusetts', 'Anaheim', 'Chicago-Sketch']
    input['selection'] = ['SiouxFalls', 'Eastern-Massachusetts', 'Anaheim']
    input['nets_dir'] = 'TransportationNetworks'  # original transport networks
    input['data_dir'] = 'DatasetsDUE'  # directory where the new datasets will be created

    solution = {}
    solution['solver'] = 'aeq'   # options: 'aeq' (fast but unstable) & 'ipp' (stable but slow) (use ipp for SiouxFalls and aeq for others)
    solution['iterations'] = 1000
    solution['tolerance'] = 1e-6
    solution['algorithm'] = 'bfw'

    variation = {}
    variation['demand_sd'] = 0.2  # standard deviation for demand variation
    variation['cur_lanes'] = 4    # assumed number of existing (current) lanes per link
    variation['new_lanes'] = 2    # max number of (new) lanes to add or change direction (swap) each link

    return dataset, input, solution, variation


# read OD matrix
def read_od(od_file):
    """
       read OD matrix
    """

    f = open(od_file, 'r')
    all_rows = f.read()
    blocks = all_rows.split('Origin')[1:]
    matrix = {}
    for k in range(len(blocks)):
        orig = blocks[k].split('\n')
        dests = orig[1:]
        origs = int(orig[0])

        d = [eval('{' + a.replace(';', ',').replace(' ', '') + '}') for a in dests]
        destinations = {}
        for i in d:
            destinations = {**destinations, **i}
        matrix[origs] = destinations
    zones = max(matrix.keys())
    od_dict = {}
    for i in range(zones):
        for j in range(zones):
            demand = matrix.get(i + 1, {}).get(j + 1, 0)
            if demand:
                od_dict[(i + 1, j + 1)] = demand
            else:
                od_dict[(i + 1, j + 1)] = 0

    return od_dict


# read network file
def read_net(net_file):
    """
       read network file
    """

    net_data = pd.read_csv(net_file, skiprows=8, sep='\t')
    # make sure all headers are lower case and without trailing spaces
    trimmed = [s.strip().lower() for s in net_data.columns]
    net_data.columns = trimmed
    # And drop the silly first and last columns
    net_data.drop(['~', ';'], axis=1, inplace=True)

    # make sure everything makes sense (otherwise some solvers throw errors)
    net_data.loc[net_data['free_flow_time'] <= 0, 'free_flow_time'] = 1e-6
    net_data.loc[net_data['capacity'] <= 0, 'capacity'] = 1e-6
    net_data.loc[net_data['length'] <= 0, 'length'] = 1e-6
    net_data.loc[net_data['power'] <= 1, 'power'] = int(4)
    net_data['init_node'] = net_data['init_node'].astype(int)
    net_data['term_node'] = net_data['term_node'].astype(int)
    net_data['b'] = net_data['b'].astype(float)

    # extract features in dict format
    links = list(zip(net_data['init_node'], net_data['term_node']))
    caps = dict(zip(links, net_data['capacity']))
    fftt = dict(zip(links, net_data['free_flow_time']))
    lent = dict(zip(links, net_data['length']))
    alpha = dict(zip(links, net_data['b']))
    beta = dict(zip(links, net_data['power']))

    net = {'capacity': caps, 'free_flow': fftt, 'length': lent, 'alpha': alpha, 'beta': beta}

    return net


# read case study data
def read_cases(selection, input_dir):
    """
       read case study data
    """

    # dictionaries for network and OD files
    net_dict = {}
    ods_dict = {}
    # selected cases
    if selection:
        cases = [case for case in selection]
    else:
        # all folders available (each one for one specific case)
        cases = [x for x in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, x))]
    # iterate through cases and read network and OD
    for case in cases:
        mod = os.path.join(input_dir, case)
        mod_files = os.listdir(mod)
        for i in mod_files:
            # read network
            if i.lower()[-8:] == 'net.tntp':
                net_file = os.path.join(mod, i)
                net_dict[case] = read_net(net_file)
            # read OD matrix
            if 'TRIPS' in i.upper() and i.lower()[-5:] == '.tntp':
                ods_file = os.path.join(mod, i)
                ods_dict[case] = read_od(ods_file)

    return net_dict, ods_dict


# solve the DUE with ipopt (convex solver)
def due_ipp(node_features, edge_features):
    """
       solve the DUE with ipopt (convex solver)
    """


    start_time = time.time()

    # read input data
    fftt = edge_features['free_flow']
    caps = edge_features['capacity']
    beta = edge_features['beta']
    alpha = edge_features['alpha']
    links = list(fftt.keys())
    nodes = np.unique([list(edge) for edge in links])
    origs = np.unique([a_node for (a_node, b_node) in list(node_features.keys())])
    dests = np.unique([b_node for (a_node, b_node) in list(node_features.keys())])

    # create node-destination demand matrix (not a regular OD!)
    demand_total = 0
    demand = {(n, d): 0 for n in nodes for d in dests}
    for r in origs:
        for s in dests:
            if (r, s) in node_features:
                demand[r, s] = node_features[r, s]
                demand_total += demand[r, s]
    for s in dests:
        demand[s, s] = - sum(demand[j, s] for j in origs)

    coef = {(i, j): fftt[i, j] * alpha[i, j] / (caps[i, j] ** beta[i, j]) for (i, j) in links}

    # create model
    model_due = pyo.ConcreteModel(name='DUE')

    # decision variables
    model_due.flow = pyo.Var([(i, j) for (i, j) in links], domain=pyo.NonNegativeReals)
    model_due.flow_destination = pyo.Var([(i, j) for (i, j) in links], [s for s in dests], domain=pyo.NonNegativeReals)

    # constraints
    model_due.conservation_flow = pyo.ConstraintList()
    for (i, j) in links:
        model_due.conservation_flow.add(sum(model_due.flow_destination[i, j, s] for s in dests)
                                        == model_due.flow[i, j])

    model_due.conservation_demand = pyo.ConstraintList()
    for i in nodes:
        for s in dests:
            model_due.conservation_demand.add(
                sum(model_due.flow_destination[i, j, s] for j in nodes if (i, j) in links) -
                sum(model_due.flow_destination[j, i, s] for j in nodes if (j, i) in links) == demand[i, s])

    # objective function
    model_due.objective = pyo.Objective(expr=(
        sum(model_due.flow[i, j] * fftt[i, j] + coef[i, j] / (beta[i, j] + 1) * ((model_due.flow[i, j]) ** (beta[i, j] + 1))
            for (i, j) in links)))

    # solve
    solver = SolverFactory('ipopt')
    result = solver.solve(model_due)
    ct_due = time.time() - start_time
    lf_due = {}
    fc_due = {}
    tt_due = -1

    # optimal DV and OF values (if optimal)
    if (result.solver.status == SolverStatus.ok) and (
            result.solver.termination_condition == TerminationCondition.optimal):
        tt_due = 0
        for (i, j) in links:
            # link flow
            lf_due[i, j] = round(pyo.value(model_due.flow[i, j]), 4)
            # flow to capacity
            fc_due[i, j] = lf_due[i, j]/caps[i, j]
            # total travel time
            tt_due += lf_due[i, j] * (fftt[i, j] + coef[i, j] * (lf_due[i, j] ** beta[i, j]))

    return lf_due, tt_due, ct_due, 1e-6


# solve DUE with equilibrae (FW, MSA, CFW, BFW)
def due_aeq(node_features, edge_features, algorithm='bfw', iterations=1000, tolerance=1e-6):
    """
       solve the DUE with equilibrae (FW, MSA, CFW, BFW)
    """

    # prep input
    fftt = edge_features['free_flow']
    caps = edge_features['capacity']
    beta = edge_features['beta']
    alpha = edge_features['alpha']
    links = list(fftt.keys())
    nodes = np.unique([list(edge) for edge in links])
    od_pairs = list(node_features.keys())
    a_nodes = [a_node for (a_node, b_node) in list(edge_features['beta'].keys())]
    b_nodes = [b_node for (a_node, b_node) in list(edge_features['beta'].keys())]
    origs = np.unique([a_node for (a_node, b_node) in od_pairs])
    dests = np.unique([b_node for (a_node, b_node) in od_pairs])
    zones = int(max(max(origs), max(dests)))
    index = np.arange(zones) + 1
    index = index.astype(int)

    # OD list to  OD matrix
    od_mat = np.zeros((zones, zones))
    for (a, b) in node_features:
        od_mat[int(a) - 1, int(b) - 1] = node_features[a, b]

    # create an AequilibraE Matrix
    demand = AequilibraeMatrix()
    kwargs = {'zones': zones,
              'matrix_names': ['matrix'],
              "memory_only": True}

    demand.create_empty(**kwargs)
    demand.matrix['matrix'][:, :] = od_mat[:, :]
    demand.index[:] = index[:].astype(int)
    demand.index = demand.index.astype(int)
    demand.computational_view(["matrix"])

    # prep network
    network = pd.DataFrame(edge_features)
    network.insert(0, 'a_node', a_nodes)
    network.insert(1, 'b_node', b_nodes)
    network = network.assign(direction=1)
    network.index = list(range(len(network)))
    network["link_id"] = network.reset_index().index + 1
    network = network.astype({"a_node": "int64", "b_node": "int64", "direction": "int64", "link_id": "int64"})

    # build graph from network
    g = Graph()
    g.cost = network['free_flow'].values
    g.capacity = network['capacity'].values
    g.free_flow = network['free_flow'].values

    # prep graph
    g.network = network
    g.network_ok = True
    g.status = 'OK'
    g.prepare_graph(index)
    g.set_graph("free_flow")
    g.cost = np.array(g.cost, copy=True)
    g.set_skimming(["free_flow"])
    g.set_blocked_centroid_flows(False)
    g.network["id"] = g.network.link_id

    # assigment using the graph g we made and aeq demand matrix
    assignclass = TrafficClass("car", g, demand)

    assign = TrafficAssignment()

    assign.set_classes([assignclass])
    assign.set_vdf("BPR")
    assign.set_vdf_parameters({"alpha": "alpha", "beta": "beta"})
    assign.set_capacity_field("capacity")
    assign.set_time_field("free_flow")
    assign.set_algorithm(algorithm)
    assign.max_iter = iterations
    assign.rgap_target = tolerance
    try:
        t_0 = time.time()
        assign.execute()
        computation_time = (time.time() - t_0)
        link_flows = dict(zip(links, list(assign.results()['matrix_ab'])))
        link_times = dict(zip(links, list(assign.results()['Congested_Time_AB'])))
        lf_array = np.array(list(assign.results()['matrix_ab']))
        lt_array = np.array(list(assign.results()['Congested_Time_AB']))
        ttt = lf_array @ lt_array
        rgap_list = list(assign.report()['rgap'])
        r_gap = rgap_list[-1]
    except:
        ttt = np.inf
        r_gap = np.inf
        link_flows = []
        computation_time = 0

    return link_flows, ttt, computation_time, r_gap


# save solution data to csv files
def due_data_save(case_dir, node_feature_data, edge_label_data, edge_feature_ff, edge_feature_cp):
    """
       save solution data to csv files
    """

    if not os.path.exists(case_dir):
        os.makedirs(case_dir)

    # clean up results
    edge_label_data = edge_label_data.dropna(how='all')
    edge_feature_ff = edge_feature_ff.dropna(how='all')
    edge_feature_cp = edge_feature_cp.dropna(how='all')
    node_feature_data = node_feature_data.dropna(how='all')

    train_range = range(0, int(len(edge_label_data) * 0.9))
    test_range = range(int(len(edge_label_data) * 0.9), int(len(edge_label_data) * 0.95))
    val_range = range(int(len(edge_label_data) * 0.95), int(len(edge_label_data)))

    # save solution data to csv files
    node_feature_data.loc[train_range].to_csv(case_dir + '/node_features_train.csv', index=False)
    node_feature_data.loc[test_range].to_csv(case_dir + '/node_features_test.csv', index=False)
    node_feature_data.loc[val_range].to_csv(case_dir + '/node_features_val.csv', index=False)

    edge_feature_ff.loc[train_range].to_csv(case_dir + '/edge_features_ff_train.csv', index=False)
    edge_feature_cp.loc[train_range].to_csv(case_dir + '/edge_features_cp_train.csv', index=False)
    edge_feature_ff.loc[test_range].to_csv(case_dir + '/edge_features_ff_test.csv', index=False)
    edge_feature_cp.loc[test_range].to_csv(case_dir + '/edge_features_cp_test.csv', index=False)
    edge_feature_ff.loc[val_range].to_csv(case_dir + '/edge_features_ff_val.csv', index=False)
    edge_feature_cp.loc[val_range].to_csv(case_dir + '/edge_features_cp_val.csv', index=False)

    edge_label_data.loc[train_range].to_csv(case_dir + '/edge_labels_train.csv', index=False)
    edge_label_data.loc[test_range].to_csv(case_dir + '/edge_labels_test.csv', index=False)
    edge_label_data.loc[val_range].to_csv(case_dir + '/edge_labels_val.csv', index=False)


# generate a dgl dataset and save to pickle
def due_dataset_pickle(case, data_dir):
    """
       generate a dgl dataset and save to pickle
    """

    # we import stuff here to make this function independent (for testing,etc.)
    import pickle
    from data_dataset_prep import DUEDatasetDGL

    # define directory and dataset
    case_dir = f'{data_dir}/{case}'
    dataset = DUEDatasetDGL(case, data_dir)

    print('Saving dataset to pickle...')
    with open(f'{case_dir}/{case}.pkl', 'wb') as f:
        pickle.dump([dataset.train, dataset.val, dataset.test], f)
    print('Done!')


# generate dataset
def generate_due_dataset():
    """
       generate a dataset with solved instances of DUE for each network in benchmark networks
       parameters are defined in parameters() and the output is saved in specified folder
    """

    # read case data
    dataset_specs, input, solution, variation = parameters()
    net_dict, ods_dict = read_cases(input['selection'], input['nets_dir'])
    data_dir = input['data_dir']
    itr = solution['iterations']
    tol = solution['tolerance']
    alg = solution['algorithm']

    # iterate through cases (networks) in selection and create variations
    for case in input['selection']:

        t_case = time.time()
        # create a new folder for case in datasets
        case_dir = f'{data_dir}/{case}'
        if not os.path.exists(case_dir):
            os.makedirs(case_dir)

        edge_features = net_dict[case]
        node_features = ods_dict[case]

        # prep dataframes to save results
        ods = list(node_features.keys())
        edges = list(edge_features['capacity'].keys())
        pairs = sorted(list(set([tuple(sorted(t)) for t in edges])))
        nodes = np.unique([list(edge) for edge in edges])
        origs = np.unique([a_node for (a_node, b_node) in list(node_features.keys())])
        dests = np.unique([b_node for (a_node, b_node) in list(node_features.keys())])

        free_flow = edge_features['free_flow']
        capacity_base = edge_features['capacity'].copy()
        edge_label_data = pd.DataFrame(columns=edges, index=range(dataset_specs['n_samples']))
        edge_feature_fftt = pd.DataFrame(columns=edges, index=range(dataset_specs['n_samples']))
        edge_feature_caps = pd.DataFrame(columns=edges, index=range(dataset_specs['n_samples']))
        node_feature_data = pd.DataFrame(columns=ods, index=range(dataset_specs['n_samples']))

        print(f'\n{case} network: generating training data and solving instance 0')
        t_0 = time.time()
        # solve base case
        if solution['solver'] == 'ipp':
            link_flow, ttt, computation_time, gap = due_ipp(node_features, edge_features)
        if solution['solver'] == 'aeq':
            link_flow, ttt, computation_time, gap = due_aeq(node_features, edge_features, alg, itr, tol)

        print('Computation time: %.2f seconds' % (time.time() - t_0))

        # save results
        edge_label_data.loc[0] = link_flow
        edge_feature_caps.loc[0] = edge_features['capacity']
        edge_feature_fftt.loc[0] = edge_features['free_flow']
        node_feature_data.loc[0] = node_features

        # the rest of the instances
        np.random.seed(dataset_specs['n_samples'])  # for reproducibility
        for i in range(1, dataset_specs['n_samples']):

            print(f'\n{case} network: generating training data for instance {i}')

            # demand variation
            node_features_i = {
                (a, b): np.amax([0.0, np.random.uniform((1 - variation['demand_sd']) * demand, (1 + variation['demand_sd']) * demand)])
                for (a, b), demand in node_features.items()}

            # lane add & remove
            cap_var = {(an, bn): np.random.randint(-1 * variation['new_lanes'], variation['new_lanes'] + 1)
                       for (an, bn) in edges}

            capacity = {(an, bn): np.max([1.0, capacity_base[(an, bn)] * (1 + (cap_var[(an, bn)]/variation['cur_lanes']))])
                        for (an, bn) in edges}

            edge_features_i = copy.deepcopy(edge_features)
            edge_features_i['capacity'] = capacity

            t_0 = time.time()
            print(f'{case} network: solving instance {i}')
            try:
                if solution['solver'] == 'ipp':
                    link_flow, ttt, computation_time, gap = due_ipp(node_features_i, edge_features_i)
                if solution['solver'] == 'aeq':
                    link_flow, ttt,  computation_time, gap = due_aeq(node_features_i, edge_features_i, alg, itr, tol)
            except:
                # not the best practice but to avoid interruptions by strange errors of aeq and ipp
                pass

            print('Solving time: %.2f seconds' % (time.time() - t_0))

            print(f'storing the results')
            if link_flow:
                edge_label_data.loc[i] = link_flow
                edge_feature_fftt.loc[i] = free_flow
                edge_feature_caps.loc[i] = capacity
                node_feature_data.loc[i] = node_features_i
            link_flow = []

            if i % 100 == 0:
                print('Saving temporary results dataset to csv...')
                # save training data as csv
                due_data_save(case_dir, node_feature_data, edge_label_data, edge_feature_fftt,
                              edge_feature_caps)

        print('Cleaning up results...')
        # clean up results
        edge_label_data = edge_label_data.dropna(how='all')
        edge_feature_ff = edge_feature_fftt.dropna(how='all')
        edge_feature_cp = edge_feature_caps.dropna(how='all')
        node_feature_data = node_feature_data.dropna(how='all')

        print('Saving dataset to csv...')
        # save training data as csv
        due_data_save(case_dir, node_feature_data, edge_label_data, edge_feature_ff, edge_feature_cp)

        # create a dgl dataset and save results as pickle
        print('Creating a dgl dataset...')
        due_dataset_pickle(case, data_dir)
        # due_dataset_pickle('Anaheim', 'DatasetsDUE')

        print(f'\ntotal time for generating a dataset with {i+1} samples of {case}:')
        print('%.2f seconds' % (time.time() - t_case))
        print('----------------------------------------------------------')
        print('----------------------------------------------------------')


if __name__ == "__main__":
    """
       just runs generate_due_dataset()
    """
    generate_due_dataset()





