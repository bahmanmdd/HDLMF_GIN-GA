"""
   Created by: Bahman Madadi
   Description: main pipeline for NDP benchmark study using deep-learning-metaheuristic framework
                It is recommended to follow the steps mentioned in the readme rather than running this.
"""


import data_due_generate as data
from train_tune import tune_bays
import train_test
import ndp_la_bm, ndp_ls_bm


def specify_steps():

    # steps to execute benchmark
    steps = {}
    steps['0_dataset_generate'] = True
    steps['1_hp_tune'] = True
    steps['2_train_test'] = True
    steps['3_benchmark'] = True

    return steps


if __name__ == "__main__":

    steps = specify_steps()

    if steps['0_dataset_generate']:
        data.generate_due_dataset()

    if steps['1_hp_tune']:
        best_params_bays = tune_bays()

    if steps['2_train_test']:
        train_test.main()

    if steps['3_benchmark']:
        ndp_la_bm
        ndp_ls_bm


