"""
    created by: Bahman Madadi
    Description: transforms training data to a format usable by DGL (for learning to solve DUE problems using GNN)
    based on the framework recommended by graphdeeplearning/benchmarking-gnns
    adapted from: https://github.com/graphdeeplearning/benchmarking-gnns/blob/master/data/TSP.py
"""

import time
import pickle
import numpy as np
import pandas as pd
import re
import itertools
import dgl
import torch
from torch.utils.data import Dataset


class DUE(Dataset):
    """
        dataset class
    """
    def __init__(self, data_dir, case, split="train", max_samples=50000):
        self.data_dir = data_dir
        self.split = split
        self.filename_nf = f'{data_dir}/{case}/node_features_{split}.csv'
        self.filename_eff = f'{data_dir}/{case}/edge_features_ff_{split}.csv'
        self.filename_ecp = f'{data_dir}/{case}/edge_features_cp_{split}.csv'
        self.filename_el = f'{data_dir}/{case}/edge_labels_{split}.csv'
        self.filename_pk = f'{data_dir}/{case}/{case}.pkl'
        self.max_samples = max_samples
        self.is_test = split.lower() in ['test', 'val']

        self.graph_lists = []
        self.edge_labels = []
        self._prepare()
        self.n_samples = len(self.edge_labels)

    def _prepare(self):
        print('\npreparing all graphs for the %s set...' % self.split.upper())

        node_feature_data = pd.read_csv(self.filename_nf, nrows=self.max_samples)
        edge_feature_cp = pd.read_csv(self.filename_ecp, nrows=self.max_samples)
        edge_feature_ff = pd.read_csv(self.filename_eff, nrows=self.max_samples)
        edge_label_data = pd.read_csv(self.filename_el, nrows=self.max_samples)

        temp = [re.findall(r"[\w']+", item) for item in node_feature_data.columns]
        ods = [[int(item[0]), int(item[1])] for item in temp]

        for graph_idx in range(len(edge_label_data)):
            # prep graph data
            edges = [[int(edge[1:-1].split(',')[0]) - 1, int(edge[1:-1].split(',')[1]) - 1] for edge in edge_label_data]
            num_nodes = len(np.unique(edges))
            num_edges = len(edges)
            demand = np.zeros((num_nodes, num_nodes))
            for od_idx in range(len(ods)):
                demand[ods[od_idx][0] - 1, ods[od_idx][1] - 1] = node_feature_data.iloc[graph_idx, od_idx]

            # Construct the DGL graph
            g = dgl.DGLGraph()
            g.add_nodes(num_nodes)
            g.ndata['feat'] = torch.Tensor(demand).float()

            edge_feats = []
            edge_labels = []

            for edge_idx in range(num_edges):
                g.add_edge(edges[edge_idx][0], edges[edge_idx][1])
                edge_feats.append([float(edge_feature_cp.loc[graph_idx][edge_idx]),
                                   float(edge_feature_ff.loc[graph_idx][edge_idx])])
                edge_labels.append(float(edge_label_data.loc[graph_idx][edge_idx]))

            # Sanity check
            assert len(edge_feats) == g.number_of_edges() == len(edge_labels)

            # Add edge features
            g.edata['feat'] = torch.Tensor(edge_feats).float()

            # # Uncomment to add dummy edge features instead (for Residual Gated ConvNet)
            # edge_feat_dim = g.ndata['feat'].shape[1] # dim same as node feature dim
            # g.edata['feat'] = torch.ones(g.number_of_edges(), edge_feat_dim)

            self.graph_lists.append(g)
            self.edge_labels.append(edge_labels)
        print('Done preparing all graphs for the %s set!' % self.split.upper())

    def __len__(self):
        """Return the number of graphs in the dataset."""
        return self.n_samples

    def __getitem__(self, idx):
        """
            Get the idx^th sample.
            Parameters
            ---------
            idx : int
                The sample index.
            Returns
            -------
            (dgl.DGLGraph, list)
                DGLGraph with node feature stored in `feat` field
                And a list of labels for each edge in the DGLGraph.
        """
        return self.graph_lists[idx], self.edge_labels[idx]


class DUEDatasetDGL(Dataset):
    """
        dataset class with case study and train/test attributes
    """

    def __init__(self, case, data_dir):
        self.case = case
        self.datadir = data_dir
        self.train = DUE(data_dir=data_dir, case=case, split="train", max_samples=50000)
        self.val = DUE(data_dir=data_dir, case=case, split="val", max_samples=50000)
        self.test = DUE(data_dir=data_dir, case=case, split="test", max_samples=50000)


class DUEDataset(Dataset):
    """
        main class for generating dgl datasets with solved instances of DUE as attributed graphs
    """

    def __init__(self, problem, network):
        start = time.time()
        self.problem = problem
        self.network = network
        self.name = f'{problem}_{network}'
        data_dir = f'Datasets{problem}/{network}'
        print(f"[I] Loading dataset {self.name}...")
        with open(f'{data_dir}/{network}.pkl', "rb") as file:
            f = pickle.load(file)
            self.train = f[0]
            self.test = f[1]
            self.val = f[2]
        print('train, test, val sizes :', len(self.train), len(self.test), len(self.val))
        print("[I] Finished loading.")
        print("[I] Data load time: {:.4f}s".format(time.time() - start))

    # form a mini batch from a given list of samples = [(graph, label) pairs]
    def collate(self, samples):
        # The input samples is a list of pairs (graph, label).
        graphs, labels = map(list, zip(*samples))
        # Edge classification labels need to be flattened to 1D lists
        labels = torch.FloatTensor(np.array(list(itertools.chain(*labels))))

        batched_graph = dgl.batch(graphs)

        return batched_graph, labels

    # prepare dense tensors for GNNs using them; such as RingGNN, 3WLGNN
    def collate_dense_gnn(self, samples, edge_feat):
        # The input samples is a list of pairs (graph, label).
        graphs, labels = map(list, zip(*samples))
        # Edge classification labels need to be flattened to 1D lists
        labels = torch.LongTensor(np.array(list(itertools.chain(*labels))))

        g = graphs[0]
        adj = self._sym_normalize_adj(g.adjacency_matrix().to_dense())
        """
            Adapted from https://github.com/leichen2018/Ring-GNN/
            Assigning node and edge feats::
            we have the adjacency matrix in R^{n x n}, the node features in R^{d_n} and edge features R^{d_e}.
            Then we build a zero-initialized tensor, say T, in R^{(1 + d_n + d_e) x n x n}. T[0, :, :] is the adjacency matrix.
            The diagonal T[1:1+d_n, i, i], i = 0 to n-1, store the node feature of node i. 
            The off diagonal T[1+d_n:, i, j] store edge features of edge(i, j).
        """

        zero_adj = torch.zeros_like(adj)

        in_node_dim = g.ndata['feat'].shape[1]
        in_edge_dim = g.edata['feat'].shape[1]

        if edge_feat:
            # use edge feats also to prepare adj
            adj_with_edge_feat = torch.stack([zero_adj for j in range(in_node_dim + in_edge_dim)])
            adj_with_edge_feat = torch.cat([adj.unsqueeze(0), adj_with_edge_feat], dim=0)

            us, vs = g.edges()
            for idx, edge_feat in enumerate(g.edata['feat']):
                adj_with_edge_feat[1 + in_node_dim:, us[idx], vs[idx]] = edge_feat

            for node, node_feat in enumerate(g.ndata['feat']):
                adj_with_edge_feat[1:1 + in_node_dim, node, node] = node_feat

            x_with_edge_feat = adj_with_edge_feat.unsqueeze(0)

            return None, x_with_edge_feat, labels, g.edges()
        else:
            # use only node feats to prepare adj
            adj_no_edge_feat = torch.stack([zero_adj for j in range(in_node_dim)])
            adj_no_edge_feat = torch.cat([adj.unsqueeze(0), adj_no_edge_feat], dim=0)

            for node, node_feat in enumerate(g.ndata['feat']):
                adj_no_edge_feat[1:1 + in_node_dim, node, node] = node_feat

            x_no_edge_feat = adj_no_edge_feat.unsqueeze(0)

            return x_no_edge_feat, None, labels, g.edges()

    def _sym_normalize_adj(self, adj):
        deg = torch.sum(adj, dim=0)  # .squeeze()
        deg_inv = torch.where(deg > 0, 1. / torch.sqrt(deg), torch.zeros(deg.size()))
        deg_inv = torch.diag(deg_inv)
        return torch.mm(deg_inv, torch.mm(adj, deg_inv))

    def _add_self_loops(self):
        """
           No self-loop support
        """
        raise NotImplementedError




