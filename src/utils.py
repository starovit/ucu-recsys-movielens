import numpy as np
from numpy.linalg import norm
import pandas as pd

def load_data():
    basic_path_to_dataset = "../../data/ml-1m/"

    # read ratings
    filename = "ratings.dat"
    df_ratings = pd.read_csv(basic_path_to_dataset+filename, header=None, sep="::")
    df_ratings.columns = ["UserID", "MovieID", "Rating", "Timestamp"]

    # read users
    filename = "users.dat"
    df_users = pd.read_csv(basic_path_to_dataset+filename, header=None, sep="::")
    df_users.columns = ["UserID", "Gender", "Age", "Occupation", "Zip-Code"]

    # read movies
    filename = "movies.dat"
    df_movies = pd.read_csv(basic_path_to_dataset+filename, header=None, sep="::")
    df_movies.columns = ["MovieID", "Title", "Genres"]

    return df_movies, df_users, df_ratings

def cosine_similarity(vector1, vector2):
    cosine = np.dot(vector1, vector2) / (norm(vector2) * norm(vector2))
    return cosine
