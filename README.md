# ucu-recsys-movielens

## Overview
This repository implements a recommendation system for the MovieLens 1M dataset. For more details about the dataset, visit [GroupLens](https://grouplens.org/datasets/movielens/1m/).

## Repository Structure
1. **artifacts/** - Contains saved model parameters.
2. **data/** - Stores the MovieLens 1M Dataset in .dat and .pickle formats.
3. **experiments/** - Jupyter notebooks for exploratory data analysis (EDA), metrics, and model testing.
4. **src/** - Python package with utilities, metrics, and model implementations.

## Set-up Instructions
### Creating the Conda Environment
To set up the Conda environment with all necessary dependencies, run the following command:
```sh
conda env create -f environment.yml
```
### With pip
If you prefer using pip, you can install requirements them with:
```sh
pip install -r requirements.txt 
```

### Installing the Project Package
To install the project package in an editable state, which allows for dynamic updates to the code without needing reinstallation, use:
```sh
python -m pip install -e src
```
#### Example usage:
Here's how to use the read_pickles function from the utilities module to load datasets:
```sh
from src.utils import read_pickles
df_movies, df_users, df_ratings = read_pickles(path_to_folder)
```
