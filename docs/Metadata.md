
# Equilibrium-Traffic-Networks

This repository contains three DGL datasets generated for the study "A hybrid deep-learning-metaheuristic framework for bi-level network design problems" by Bahman Madadi and Gonçalo H. de Almeida Correia, published in Expert Systems with Applications. The datasets are generated and used to train and evaluate models for solving the User Equilibrium (UE) problem on three transportation networks (Sioux-Falls, Eastern-Massachusetts, and Anaheim) from the well-known "transport networks for research" repository.

## Metadata

| Network               | Nodes | Edges | OD Pairs | Train Samples | Val Samples | Test Samples | Dataset Size | Solvers | Algorithm |
|-----------------------|-------|-------|----------|---------------|-------------|--------------|--------------|---------|-----------|
| SiouxFalls            | 24    | 76    | 576      | 18000         | 1000        | 1000         | 20,000       | Aeq, Ipp| BFW       |
| Eastern-Massachusetts | 74    | 258   | 5476     | 4000          | 500         | 500          | 5,000        | Aeq, Ipp| BFW       |
| Anaheim               | 416   | 914   | 1444     | 4000          | 500         | 500          | 5,000        | Aeq, Ipp| BFW       |

### Features and Data Fields

| Field                | Type   | Description                                      |
|----------------------|--------|--------------------------------------------------|
| Node Features        | Array  | Represent origin-destination (OD) demand matrices for travel. Each OD pair specifies travel demand between zones. |
| Edge Features        | Array  | Include: Free-flow travel time (FFTT) and Capacity. |
| Edge Labels          | Array  | Optimal link flows derived from solving the DUE problem. |
| Number of Links      | Int    | Number of links in the network.                  |
| Number of Nodes      | Int    | Number of nodes in the network.                  |
| Number of OD Pairs   | Int    | Number of origin-destination pairs in the network. |
| Train Split          | Int    | Number of samples in the training set.           |
| Validation Split     | Int    | Number of samples in the validation set.         |
| Test Split           | Int    | Number of samples in the test set.               |
| Dataset Size         | Int    | Total number of samples in the dataset.          |
| Solvers              | String | Solvers used for generating the dataset.         |
| Algorithm            | String | Algorithm used for generating the dataset.       |

## Datasets

The datasets are generated using the scripts `data_due_generate.py` and `data_dataset_prep.py` from the GitHub repository. Each dataset corresponds to a different transportation network and contains solved instances of the DUE problem.

### Dataset Structure

Each dataset is stored as a pickle file and contains three splits: train, validation, and test. Each split is a list of DGLGraph objects with node and edge features, along with edge labels.

- **Node Features**: Represent origin-destination (OD) demand matrices for travel. Each OD pair specifies travel demand between zones. Stored in the `feat` field of the DGLGraph.
- **Edge Features**: Include Free-flow travel time (FFTT) and Capacity. Stored in the `feat` field of the DGLGraph.
- **Edge Labels**: Optimal link flows derived from solving the DUE problem. A list of labels for each edge in the DGLGraph.

### Available Networks

- SiouxFalls
- Eastern-Massachusetts
- Anaheim

## Usage

To load a dataset, use the following code:

```python
import pickle
from data_dataset_prep import DUEDatasetDGL

case = 'SiouxFalls'  # or 'Eastern-Massachusetts', 'Anaheim'
data_dir = 'DatasetsDUE'

with open(f'{data_dir}/{case}/{case}.pkl', 'rb') as f:
    train, val, test = pickle.load(f)

# Example: Accessing the first graph and its edge labels in the training set
graph, edge_labels = train[0]
print(graph)
print(edge_labels)
```

## Data Generation Steps

1. **Define Parameters**: Set the parameters for dataset generation in the `parameters()` function.
2. **Solve DUE Problem**: Use the `data_due_generate.py` script to solve the DUE problem for each network.
3. **Store Results**: Save the results as CSV files and clean up the data.
4. **Create DGL Dataset**: Convert the data into DGL format using the `data_dataset_prep.py` script and save it as pickle files.

## Scripts

### `data_due_generate.py` (from GitHub repository)

This script generates the datasets by solving the DUE problem for each network in the benchmark networks. The parameters for dataset generation are defined in the `parameters()` function.

### `data_dataset_prep.py` (from GitHub repository)

This script contains the classes and functions for preparing the datasets and converting them into DGL format.

## References

- [A hybrid deep-learning-metaheuristic framework for bi-level network design problems](https://doi.org/10.1016/j.eswa.2023.122814)
- [GitHub Repository: HDLMF_GIN-GA](https://github.com/bahmanmdd/HDLMF_GIN-GA)

## Citation

If you use these datasets in your research, please cite the following paper:

Madadi B, de Almeida Correia GH. A hybrid deep-learning-metaheuristic framework for bi-level network design problems. Expert Systems with Applications. 2024 Jun 1;243:122814.

---

### Metadata

```json
{
  "datasets": [
    {
      "name": "SiouxFalls",
      "description": "DGL dataset for the SiouxFalls transportation network with solved instances of the DUE problem.",
      "num_samples": 20000,
      "features": {
        "node_features": "Represent origin-destination (OD) demand matrices for travel. Each OD pair specifies travel demand between zones.",
        "edge_features": "Include Free-flow travel time (FFTT) and Capacity.",
        "edge_labels": "Optimal link flows derived from solving the DUE problem."
      },
      "splits": ["train", "val", "test"]
    },
    {
      "name": "Eastern-Massachusetts",
      "description": "DGL dataset for the Eastern-Massachusetts transportation network with solved instances of the DUE problem.",
      "num_samples": 5000,
      "features": {
        "node_features": "Represent origin-destination (OD) demand matrices for travel. Each OD pair specifies travel demand between zones.",
        "edge_features": "Include Free-flow travel time (FFTT) and Capacity.",
        "edge_labels": "Optimal link flows derived from solving the DUE problem."
      },
      "splits": ["train", "val", "test"]
    },
    {
      "name": "Anaheim",
      "description": "DGL dataset for the Anaheim transportation network with solved instances of the DUE problem.",
      "num_samples": 5000,
      "features": {
        "node_features": "Represent origin-destination (OD) demand matrices for travel. Each OD pair specifies travel demand between zones.",
        "edge_features": "Include Free-flow travel time (FFTT) and Capacity.",
        "edge_labels": "Optimal link flows derived from solving the DUE problem."
      },
      "splits": ["train", "val", "test"]
    }
  ],
  "references": [
    {
      "title": "A hybrid deep-learning-metaheuristic framework for bi-level network design problems",
      "doi": "10.1016/j.eswa.2023.122814",
      "url": "https://doi.org/10.1016/j.eswa.2023.122814",
      "authors": ["Bahman Madadi", "Gonçalo H. de Almeida Correia"],
      "journal": "Expert Systems with Applications",
      "year": 2024,
      "volume": 243,
      "pages": "122814"
    },
    {
      "title": "GitHub Repository: HDLMF_GIN-GA",
      "url": "https://github.com/bahmanmdd/HDLMF_GIN-GA"
    }
  ]
}
```