
This repository is related to the following article entitled "A hybrid deep-learning-metaheuristic framework for bi-level network design problems" 
[Madadi and Correia (2023)](https://arxiv.org/abs/2303.06024)

For theory and methodology details, please refer to the article.

The reproducability of the results is verified by codeocean via the following computation capsule based on this repository:
[A hybrid deep-learning-metaheuristic framework for bi-level network design problems (GIN-GA23)](https://doi.org/10.24433/CO.0943845.v1)


## Setup

1. Seting up the environment
   * Setup an environment using the `requirements.txt` file. This is a pip-friendly list of the high-level python packages required to setup an environment for this project including the version information. So simply create an empty environment and use the command `pip install -r requirements.txt` to setup the environment. But before installing packages, please check the "license requirements" section below.
   * Note that if you want the full functionality, you need to acquire a CPLEX license and setup CPLEX first. See more details below under "License requirements" heading.
2. License requirements 
   * To fully utilize this repository and reproduce the experiments, you will require a CPLEX license (academics can acquire an academic license for free). See instructions [here](https://www.ibm.com/docs/en/icos/22.1.1?topic=cplex-setting-up-python-api). However, this is only necessary for implementing the SORB method. For the experiments reported in [Madadi and Correia (2023)](https://arxiv.org/abs/2303.06024), we have provided the SORB method results in csv files in the output directory here and the benchmarking code allows running other methods and benchmarking against the saved results of SORB.

3. Datasets
   * Generate: You can reproduce the datasets used for experiments in [Madadi and Correia (2023)](https://arxiv.org/abs/2303.06024) or generate new datasets using `data_due_generate` (instructions below).
   * Download: Alternatively you can just download the datasets from [this repository](https://doi.org/10.6084/m9.figshare.27889251.v1) with the comprehensive metadata.
   * Dataset citation: Madadi, Bahman (2024). Equilibrium-Traffic-Networks. figshare. Dataset. https://doi.org/10.6084/m9.figshare.27889251.v1
5. Transport networks 
   * Transport networks used for creating datasets are stored in 'TransportationNetworks' directory and are selected from [here](https://github.com/bstabler/TransportationNetworks).
6. Config files
   * Hyperparameters (for training GNNs) are saved in config files in the "configs" directory for each network and model, but you can change them. 



## Generating datasets

Use `data_due_generate` to generate new datasets with solved instances of DUE problems. 
1. In `parameters` (the first function), specify the following parameters (current values reproduce [Madadi and Correia (2023)](https://arxiv.org/abs/2303.06024)):
   1. dataset params (size, etc.) (note that different dataset sizes are used for different networks, for details refer to [Madadi and Correia (2023)](https://arxiv.org/abs/2303.06024))
   2. problem params (network, etc.)
   3. solution params (solver choice and options)
   4. variation params (perturbations for problem variation generation)
2. Run `data_due_generate`. Be aware that this might take days or weeks depending on the size.
3. Check the "DadatsetsDUE/network" directory for results.
 

### Notes:

* It is recommended to generate and prep datasets one at a time to avoid memory and other issues.
* You can solve DUE instances with: 
  1. ipopt (open source general solver), which is very reliable but slow (recommended for small networks), particularly for larger networks (e.g., Anaheim)
  2. aequilibriae (specialized solver for DUE), which is much faster (recommended for large networks) but less stable, so it might throw unpredictable errors. This means with this solver, sometimes you might have to run the code a few times to have a complete dataset.


## Train-test pipeline

* Hyperparameter tuning:
  * You can use the `train_tune` submodule for hyperparameter-tuning but the good parameters are already identified and saved in config files.
* Using the `train_test` submodule for training:
  1. Specify the `model` (e.g., GIN), the `problem` (e.g., DUE), and the `network` (e.g., Anaheim) in `problem_spec` (the first function).
  2. Hyperparameters are saved in config files in the "configs" directory for each network and model, but you can change them.
  3. Run `train_test` and check out the "output/DUE" directory for training results
  4. Best models are copied in the "models" directory for inference (from "output/DUE/network/models") but if you train better models, you can replace them.


## Benchmarking solutions

Use `ndp_la_bm` & `ndp_ls_bm` for benchmarking NDP-LA and NDP-LS problems respectively. They have a very similar structure except for minor parameter differences based on each problem.

Just define scenario and variation parameters in `scenarios` (the first function) and run. Parameters for each method are specified in the first function in `problem_method` submodules (e.g., `ndp_la_sorb`). Current values reproduce the experiments in [Madadi and Correia (2023)](https://arxiv.org/abs/2303.06024).

The output for each case study (network) will be saved in "output/problem/network". There will be a summary and a separate directory for detailed results of each solution method.

I recommend running the benchmarks one case study at a time to avoid any issue.


## Code implementation order

1. Dataset preparation with `data_due_generate`
   1. Generate datasets
   2. Convert datasets to dgl format (done within the same module)
2. Training with `train_test`
   1. Hyperparameter tuning (optional)
   2. Train-test pipeline
3. Computational experiemnts with `ndp_la_bm` & `ndp_ls_bm` (separately)
   1. Benchmark solutions
      1. SORB
      2. GIN-GA

## Citation

If you use the code and data in your research, please cite the following paper and dataset:

Paper citation: 

Madadi B, de Almeida Correia GH. A hybrid deep-learning-metaheuristic framework for bi-level network design problems. Expert Systems with Applications. 2024 Jun 1;243:122814. https://doi.org/10.1016/j.eswa.2023.122814

Dataset citation: 

Madadi, Bahman (2024). Equilibrium-Traffic-Networks. figshare. Dataset. https://doi.org/10.6084/m9.figshare.27889251.v1

