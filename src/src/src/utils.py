import numpy as np
import pandas as pd

from numpy.linalg import norm

def read_pickles(path_to_folder):
    df_movies = pd.read_pickle(path_to_folder+"movies.pickle")
    df_users = pd.read_pickle(path_to_folder+"users.pickle")
    df_ratings = pd.read_pickle(path_to_folder+"ratings.pickle")
    return df_movies, df_users, df_ratings

def read_dats(path_to_folder):
    pass

class TrainTestSplitter():
    def __init__():
        pass

def cosine_similarity(vector1, vector2):
    cosine = np.dot(vector1, vector2) / (norm(vector2) * norm(vector2))
    return cosine
