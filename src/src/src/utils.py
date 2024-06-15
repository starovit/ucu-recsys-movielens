import pandas as pd
import numpy as np
from numpy.linalg import norm

def read_pickles(path_to_folder):
    df_movies = pd.read_pickle(path_to_folder+"movies.pickle")
    df_users = pd.read_pickle(path_to_folder+"users.pickle")
    df_ratings = pd.read_pickle(path_to_folder+"ratings.pickle")
    return df_movies, df_users, df_ratings

def read_dats(path_to_folder="../../data/ml-1m/"):
    # read ratings
    filename = "ratings.dat"
    df_ratings = pd.read_csv(path_to_folder+filename, header=None, sep="::")
    df_ratings.columns = ["UserID", "MovieID", "Rating", "Timestamp"]
    # read users
    filename = "users.dat"
    df_users = pd.read_csv(path_to_folder+filename, header=None, sep="::")
    df_users.columns = ["UserID", "Gender", "Age", "Occupation", "Zip-Code"]
    # read movies
    filename = "movies.dat"
    df_movies = pd.read_csv(path_to_folder+filename, header=None, sep="::")
    df_movies.columns = ["MovieID", "Title", "Genres"]
    return df_movies, df_users, df_ratings

class TrainTestSplitter():
    def __init__():
        pass

def cosine_similarity(vector1, vector2):
    cosine = np.dot(vector1, vector2) / (norm(vector2) * norm(vector2))
    return cosine
