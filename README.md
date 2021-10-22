# Active Surrogate Estimators: Efficient Estimation of High–Dimensional Expectations

Hi, good to see you here! 👋

This is code for the anonymous submission `Active Surrogate Estimators: Efficient Estimation of High-Dimensional Expectations' to AISTATS 2021.


## Setup

We recommend you set up a conda environment like so:

```
conda-env update -f slurm/environment.yaml
conda activate ase
cd bayesquad
python setup.py install
cd ..
```

After this, you should be able to reproduce the experiments of this paper


## Reproducing the Experiments

* The `reproduce' folder contains scripts for running specific experiments.
* Execute a script as
```
sh reproduce/<script-name>.sh
```
* For most experiments, it's recommended to start the same script multiple times, e.g. across different compute nodes of the cluster. Each of the independent runs will log into it's own directory, and results will automatically combined later (as long as they're logging to the same `output' folder). (This is not needed for Fig. 5 of the paper.)
* You can then create plots with the Jupyter Notebook at
```
notebooks/plots_paper.ipynb
```
* All runs log continuously, so you should be able to create plots as the results are coming in.


* Some other notes
    * Realistically speaking, most results will require you to use a gpu.
    * If you want to understand the code, below we give a good strategy for approaching it. (Also the synthetic experiments have somewhat less complex code and are probably better to look at first.)


### Details

* To recreate the synthetic experiments of Section 5 of the paper, run the scripts `reproduce/SyntheticGaussian.sh` and `reproduce/SyntheticGaussianBayesQuad.sh`.
* To recreate the distribution shift experiments of Section 6.2 of the paper, run the script `reproduce/RadialBNNMNISTMissing7.sh`.
* To recreate the ResNet experiments of Section 6.3 of the paper, run the scripts `reproduce/ResNetCifar10.sh`, `reproduce/ResNetCifar100.sh`, and `reproduce/ResNetFMNIST.sh`.
* To recreate the additional synthetic experiment of Appendix B of the paper, run the script `reproduce/SyntheticLinear.sh`.

* Note that, as soon as the respective runs have logged more than a few runs, you can already have a look at the results by simply executing the respective cells in the `notebooks/plots_paper.ipynb` notebook.


## Running A Custom Experiment

* `main.py` is the main entry point into this code-base.
    * It executes a a total of  `n_runs` experiments for a fixed setup.
    * Each experiment:
        * Trains (or loads) one main model.
        * This model can then be evaluated with a variety of acquisition strategies.
        * Risk estimates are then computed for points/weights from all acquisition strategies for all risk estimators.
    * The code is largely framed in terms of active testing. However, we also perform general estimation of expectations with this codebase. For these scenarios, we do not use a main model (because there is no model to evaluate), and the 'risk' becomes the true value of the integral, and so on.


* This repository uses `Hydra` to manage configs.
    * Look at `conf/config.yaml` or one of the experiments in `conf/...` for default configs and hyperparameters.
    * Experiments are autologged and results saved to `./output/`.

* A guide to the code
    * `main.py` runs repeated experiments and orchestrates the whole shebang.
        * It iterates through all `n_runs` and `acquisition strategies`.
    * `experiment.py` handles a single experiment.
        * It combines the `model`, `dataset`, `acquisition strategy`, and `risk estimators`.
    * `datasets.py`, `aquisition.py`, `loss.py`, `risk_estimators.py`. Those should all contain more or less what you would expect.
    * `hoover.py` is a logging module.
    * `models/` contains all models, scikit-learn and pyTorch.
        * In `sk2torch.py` we have some code that wraps torch models in a way that lets them be used as scikit-learn models from the outside.

## And Finally

Thanks for stopping by!

If you find anything wrong with the code, please contact us.

We are happy to answer any questions related to the code and project.