# ucu-recsys-movielens
Recomendation System for Movie Lens 1M Dataset:
https://grouplens.org/datasets/movielens/1m/


## Creating the Conda Environment
To create the Conda environment from the configuration file, run the following command:

```sh
conda env create -f environment.yml
```

## Install src:
```sh
python -m pip install -e src
```
Example usage:
```sh
from src.utils import cosine_similarity
```
