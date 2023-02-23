# Introduction

This project contains the code for the paper titled
"Evaluating algorithms for probabilistic forecasting and uncertainty estimation in regression problems".

It contains the implementation of several uncertainty estimation models for application in a retail use-case,
such as LightGBM, LightGBM Quantile Regressor, Bootstrapping, NGBoost, Probabilistic Gradient Boosting Machines (PGBM),
Level Set Forecaster (LSF), Conformalized Regression, Temporal Fusion Transformer (TFT) and MQ-CNN.

Moreover, under the notebooks folder you can find the notebooks we used to train and evaluate our models on Kaggle datasets.
The Kaggle datasets we used for evaluation are the following:

1. Blue Book For Bulldozers (bulldozer)
2. Rossmann Store Sales (rossmann)
3. Corporaci´on Favorita Grocery Sales Forecasting (favorita)

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

- Windows users, for PGBM: you may need to install Build Tools for Visual Studio and
make sure you add compiler cl to your PATH environment variable (see here).
Verify that Windows can find cl by executing where cl in a Windows command line terminal.

# Build and Test

Navigate to the directory you want the git repo to live in, then run

````commandline
git clone https://github.com/daroczisandor/uncertainty_estimation.git
cd uncertainty_estimation
````

Obtain data from google drive

Store it in a folder called "datasets" at the top level of this repo
(this is where the notebooks point to when reading data).

TODO: Describe and show how to build your code and run the tests.

Datasets can be found in google drive under this link:

https://drive.google.com/drive/folders/1WV-z19PntL_PhDEwZbvPhI7WOZdxfYrO?usp=sharing

# Contribute

TODO: Explain how other users and developers can contribute to make your code better.

If you want to learn more about creating good readme files then refer the
following [guidelines](https://docs.microsoft.com/en-us/azure/devops/repos/git/create-a-readme?view=azure-devops). You
can also seek inspiration from the below readme files:

- [ASP.NET Core](https://github.com/aspnet/Home)
- [Visual Studio Code](https://github.com/Microsoft/vscode)
- [Chakra Core](https://github.com/Microsoft/ChakraCore)