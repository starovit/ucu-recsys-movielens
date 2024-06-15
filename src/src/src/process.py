import pandas as pd

def read_pickles(path_to_folder):
    df_movies = pd.read_pickle(path_to_folder+"movies.pickle")
    df_users = pd.read_pickle(path_to_folder+"users.pickle")
    df_ratings = pd.read_pickle(path_to_folder+"ratings.pickle")
    return df_movies, df_users, df_ratings


class TrainTestSplitter():
    def __init__():
        pass