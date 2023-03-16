# Introduction

This project contains the code for the paper titled
"Evaluating algorithms for probabilistic forecasting and uncertainty estimation in regression problems".

It contains the implementation of several uncertainty estimation models for application in a retail use-case,
such as LightGBM, LightGBM Quantile Regressor, Bootstrapping, NGBoost, Probabilistic Gradient Boosting Machines (PGBM),
Level Set Forecaster (LSF), Conformalized Regression, Temporal Fusion Transformer (TFT) and MQ-CNN.

Moreover, under the notebooks folder you can find the python notebooks we used to train and evaluate our models on Kaggle datasets.
The Kaggle datasets we used for the evaluation are the following:

1. Blue Book For Bulldozers (bulldozer)
2. Rossmann Store Sales (rossmann)
3. Corporación Favorita Grocery Sales Forecasting (favorita)

For more details, read our paper here.

# Getting Started

## 1. Installation

Currently, the package installation can be done only via poetry.
Read [here](https://python-poetry.org/docs/) about how to install and set up poetry on your local machine.

After you have cloned the project and set up poetry, create a virtual environment and install packages from the pyproject.toml file via

````commandline
poetry update
poetry install
````

Then, build the project using

````commandline
poetry build
````

After that, 

````commandline
pip install uncertainty_estimation_models-0.1.0-py3-none-any.whl
````

First, make sure you have separately installed the various dependencies (see below).

### Dependencies

- For Windows users having issues to install the
[PGBM](https://pgbm.readthedocs.io/en/latest/index.html) package: you may need to install
[Build Tools for Visual Studio](https://visualstudio.microsoft.com/de/downloads/) and
make sure you add compiler `cl` to your `PATH` environment variable ([see here](https://stackoverflow.com/questions/84404/using-visual-studios-cl-from-a-normal-command-line/65812244#65812244)).
Verify that Windows can find `cl` by executing `where cl` in a Windows command line terminal.

# Build and Test

Navigate to the directory you want the git repo to live in, then run

````commandline
git clone https://github.com/daroczisandor/uncertainty-estimation.git
cd uncertainty_estimation
````

Obtain data from Google Drive

Store it in a folder called "datasets" at the top level of this repo
(this is where the notebooks point to when reading data).

Datasets can be found in Google Drive under this link:

https://drive.google.com/drive/folders/1WV-z19PntL_PhDEwZbvPhI7WOZdxfYrO?usp=sharing



# Contribute

