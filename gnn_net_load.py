"""
    Utility file to select GraphNN model as
    selected by the user
"""

from gnn_net_gin import GINNet


def GIN(net_params):
    """
    Utility function to select GraphNN model as
    selected by the user
    """
    return GINNet(net_params)


def gnn_model(MODEL_NAME, net_params):
    """
    utility function for future developments with multiple GNNs
    """
    models = {
        'GIN': GIN,
    }

    return models[MODEL_NAME](net_params)
