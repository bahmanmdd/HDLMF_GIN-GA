"""
    Utility functions for training one epoch
    and evaluating one epoch
"""
import torch
import torch.nn as nn
import math
import dgl
import numpy as np
from torchmetrics import MeanAbsolutePercentageError
from torchmetrics import R2Score


def train_epoch(model, optimizer, device, data_loader, epoch):
    """
        Utility function for training one epoch
    """

    model.train()
    epoch_r2 = 0
    epoch_ttt = 0
    epoch_mape = 0
    epoch_loss = 0
    r2score = R2Score()
    mape = MeanAbsolutePercentageError()
    for iter, (batch_graphs, batch_labels) in enumerate(data_loader):
        batch_graphs = batch_graphs.to(device)
        batch_x = batch_graphs.ndata['feat'].to(device)  # num x feat
        batch_e = batch_graphs.edata['feat'].to(device)
        batch_labels = batch_labels.unsqueeze(-1).to(device)
        optimizer.zero_grad()

        batch_scores = model.forward(batch_graphs, batch_x, batch_e)

        # link flow labels
        ltt_score = batch_e[:, 1] * (1 + 0.15 * torch.pow(torch.div(torch.flatten(batch_scores), batch_e[:, 0]), 4))
        ltt_label = batch_e[:, 1] * (1 + 0.15 * torch.pow(torch.div(torch.flatten(batch_labels), batch_e[:, 0]), 4))

        n_edges = int(len(ltt_score)/batch_graphs.batch_size)
        ttt_score = torch.from_numpy(np.fromiter((ltt_score[b*n_edges:(b+1)*n_edges] @ batch_scores[b*n_edges:(b+1)*n_edges]
                                                  for b in range(batch_graphs.batch_size)), 'float'))
        ttt_label = torch.from_numpy(np.fromiter((ltt_label[b*n_edges:(b+1)*n_edges] @ batch_labels[b*n_edges:(b+1)*n_edges]
                                                  for b in range(batch_graphs.batch_size)), 'float'))

        batch_r2 = r2score(batch_scores, batch_labels)
        batch_mape = mape(batch_scores, batch_labels)
        batch_ttt_mape = mape(ttt_score, ttt_label)
        loss = model.loss(batch_scores, batch_labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.detach().item()
        epoch_mape += batch_mape.detach().item()
        epoch_ttt += batch_ttt_mape.detach().item()
        epoch_r2 += batch_r2.detach().item()
    epoch_loss /= (iter + 1)
    epoch_mape /= (iter + 1)
    epoch_ttt /= (iter + 1)
    epoch_r2 /= (iter + 1)

    return epoch_loss, epoch_mape, epoch_ttt, epoch_r2, optimizer


def evaluate_network(model, device, data_loader, epoch):
    """
        Utility functions for evaluating one epoch
    """

    model.eval()
    epoch_test_r2 = 0
    epoch_test_ttt = 0
    epoch_test_mape = 0
    epoch_test_loss = 0
    r2score = R2Score()
    mape = MeanAbsolutePercentageError()
    with torch.no_grad():
        for iter, (batch_graphs, batch_labels) in enumerate(data_loader):
            batch_graphs = batch_graphs.to(device)
            batch_x = batch_graphs.ndata['feat'].to(device)
            batch_e = batch_graphs.edata['feat'].to(device)
            batch_labels = batch_labels.unsqueeze(-1).to(device)

            batch_scores = model.forward(batch_graphs, batch_x, batch_e)

            # link flow labels
            ltt_score = batch_e[:, 1] * (1 + 0.15 * torch.pow(torch.div(torch.flatten(batch_scores), batch_e[:, 0]), 4))
            ltt_label = batch_e[:, 1] * (1 + 0.15 * torch.pow(torch.div(torch.flatten(batch_labels), batch_e[:, 0]), 4))

            n_edges = int(len(ltt_score) / batch_graphs.batch_size)
            ttt_score = torch.from_numpy(
                np.fromiter((ltt_score[b * n_edges:(b + 1) * n_edges] @ batch_scores[b * n_edges:(b + 1) * n_edges]
                             for b in range(batch_graphs.batch_size)), 'float'))
            ttt_label = torch.from_numpy(
                np.fromiter((ltt_label[b * n_edges:(b + 1) * n_edges] @ batch_labels[b * n_edges:(b + 1) * n_edges]
                             for b in range(batch_graphs.batch_size)), 'float'))

            batch_r2 = r2score(batch_scores, batch_labels)
            batch_mape = mape(batch_scores, batch_labels)
            batch_ttt_mape = mape(ttt_score, ttt_label)
            loss = model.loss(batch_scores, batch_labels)
            epoch_test_loss += loss.detach().item()
            epoch_test_mape += batch_mape.detach().item()
            epoch_test_ttt += batch_ttt_mape.detach().item()
            epoch_test_r2 += batch_r2.detach().item()
        epoch_test_loss /= (iter + 1)
        epoch_test_mape /= (iter + 1)
        epoch_test_ttt /= (iter + 1)
        epoch_test_r2 /= (iter + 1)

    return epoch_test_loss, epoch_test_mape, epoch_test_ttt, epoch_test_r2, batch_labels, batch_scores


