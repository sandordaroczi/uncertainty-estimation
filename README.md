# Uncertainty Estimation in Regression Problems

Welcome to the repository for our collaborative project, focusing on implementing various uncertainty estimation models tailored for a retail use-case. The project includes models like LightGBM, LightGBM Quantile Regressor, Bootstrapping, NGBoost, Probabilistic Gradient Boosting Machines (PGBM), Level Set Forecaster (LSF), Conformalized Regression, Temporal Fusion Transformer (TFT), and MQ-CNN.

## Notebooks and Kaggle Datasets
Under the notebooks folder, you'll find Python notebooks utilized to train and evaluate the models on Kaggle datasets. The Kaggle datasets used for evaluation include:

1. Blue Book For Bulldozers (bulldozer)
2. Rossmann Store Sales (rossmann)
3. Corporación Favorita Grocery Sales Forecasting (favorita)

## Getting Started

### 1. Installation

Ensure you have [poetry](https://python-poetry.org/docs/) installed on your local machine. After cloning the project, create a virtual environment and install packages from the pyproject.toml file:

````commandline
poetry update
poetry install
poetry build
pip install uncertainty_estimation_models-0.1.0-py3-none-any.whl
````

Make sure to separately install the various dependencies listed below.

#### Dependencies

For Windows users encountering issues installing the [PGBM](https://pgbm.readthedocs.io/en/latest/index.html) package, install [Build Tools for Visual Studio](https://visualstudio.microsoft.com/de/downloads/) and ensure you add the compiler cl to your PATH environment variable ([see here](https://stackoverflow.com/questions/84404/using-visual-studios-cl-from-a-normal-command-line/65812244#65812244)). Verify Windows can find cl by executing where cl in a Windows command line terminal.

### 2. Build and Test

Navigate to the desired directory for the git repo and run:

````commandline
git clone https://github.com/daroczisandor/uncertainty-estimation.git
cd uncertainty_estimation
````

Obtain the required datasets from [Google Drive](https://drive.google.com/drive/folders/1WV-z19PntL_PhDEwZbvPhI7WOZdxfYrO?usp=sharing) and store them in a folder called "datasets" at the top level of this repo.

## Contribute

Contributions are welcome! Feel free to create issues or submit pull requests to enhance and extend the project. 🚀
