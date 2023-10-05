"""
    visualization of results of training and test results
    created by: Bahman Madadi
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from hampel import hampel

plt.style.use('default')


def plot_results(train_data, test_data, criterion, name):

    train_data = hampel(pd.Series(train_data), window_size=5, n=5, imputation=True)
    test_data = hampel(pd.Series(test_data), window_size=5, n=5, imputation=True)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    ax.set_xlabel('Epoch')
    ax.set_ylabel(f'{criterion}')

    ax.tick_params(axis='both', which='both')

    line_test = ax.plot(test_data, linewidth=1)
    line_test[0].set_label('Test')

    line_train = ax.plot(train_data, linewidth=1.2)
    line_train[0].set_label('Training')

    # if "MSE" not in criterion:
    #     ax.set_ylim(y_min, y_max)
    ax.legend()
    ax.tick_params(axis='both', which='both', labelsize=7)
    # ax.grid(True, which='both')
    # plt.minorticks_on()

    fig.savefig(f'{name}{criterion}.jpeg', dpi=600)

    plt.close("all")

