import pandas as pd
import numpy as np
from numpy.linalg import norm

def read_pickles(path_to_folder):
    df_movies = pd.read_pickle(path_to_folder+"movies.pickle")
    df_users = pd.read_pickle(path_to_folder+"users.pickle")
    df_ratings = pd.read_pickle(path_to_folder+"ratings.pickle")
    return df_movies, df_users, df_ratings

def read_dats(path_to_folder="../../data/ml-1m/"):
    df_ratings = pd.read_csv(path_to_folder+"ratings.dat", header=None, sep="::")
    df_users = pd.read_csv(path_to_folder+"users.dat", header=None, sep="::")
    df_movies = pd.read_csv(path_to_folder+"movies.dat", header=None, sep="::")
    df_ratings.columns = ["UserID", "MovieID", "Rating", "Timestamp"]
    df_users.columns = ["UserID", "Gender", "Age", "Occupation", "Zip-Code"]
    df_movies.columns = ["MovieID", "Title", "Genres"]
    return df_movies, df_users, df_ratings

class TrainTestSplitter:
    """Handles splitting of dataframes into train and test sets."""
    def __init__(self):
        pass

    @classmethod
    def split_by_date(self, df, split_date):
        train = df[df["Date"] < split_date]
        test = df[df["Date"] >= split_date]
        return train, test

    @classmethod
    def split_by_percent(self, df, percent=0.8, random_split=False):
        if random_split:
            df = df.sample(frac=1)
        split_index = int(df.shape[0] * percent)
        train = df.iloc[:split_index]
        test = df.iloc[split_index:]
        return train, test


def cosine_similarity(vector1, vector2):
    cosine = np.dot(vector1, vector2) / (norm(vector2) * norm(vector2))
    return cosine
