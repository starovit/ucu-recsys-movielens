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
    def split_by_percent(self, df, percent=0.8, random_split=False,
                         sort_by_datetime=False):
        df = df.copy()
        if sort_by_datetime:
            df = df.sort_values("Datetime")
        if random_split:
            df = df.sample(frac=1, random_state=1)
        split_index = int(df.shape[0] * percent)
        train = df.iloc[:split_index]
        test = df.iloc[split_index:]
        return train, test
    
    @classmethod
    def split_by_users(self, df, n_reviews_in_test=10):
        train_list, test_list = zip(*df.groupby('UserID').apply(self.split_user_reviews, n_reviews_in_test))
        # Concatenate the list of DataFrames into a single DataFrame for train and test
        train = pd.concat([x for x in train_list if x is not None])
        test = pd.concat([x for x in test_list if x is not None])
        return train, test


    def split_user_reviews(group, n_reviews_in_test):
        if len(group) > n_reviews_in_test:
            return group.iloc[:-10], group.iloc[-10:]
        else:
            return group, None


def cosine_similarity(vector1, vector2):
    cosine = np.dot(vector1, vector2) / (norm(vector2) * norm(vector2))
    return cosine



# deep-learning data-prep functions

def map_year_to_decade(year):
    try:
        year = int(year)
        if 1900 <= year <= 2000:
            return f"{(year // 10) * 10}"
        else:
            return "Other"
    except:
        return "Other"
    
def compute_rolling_averages(group):
    return group.expanding().mean().shift()
    
def dl_data_pipeline(df_movies, df_users, df_ratings):
    df_users = df_users[["UserID", "Gender", "Age", "Occupation", "State"]]
    categorical_features = ['Gender', 'Age', 'Occupation', 'State']
    df_users = pd.get_dummies(df_users, columns=categorical_features, dtype=int)

    df_movies = df_movies.drop(["Title", "Genres"], axis=1)
    df_movies['Decade'] = df_movies['Year'].apply(map_year_to_decade)
    df_movies = pd.get_dummies(df_movies, columns=['Decade'])
    df_movies = df_movies.drop(["Year"], axis=1).astype(int)

    df_ratings = df_ratings[["UserID", "MovieID", "Rating", "Datetime"]]
    df_ratings = df_ratings.sort_values("Datetime")
    df_ratings['AvgUserRating'] = df_ratings.groupby('UserID')['Rating'].transform(compute_rolling_averages)
    df_ratings['AvgMovieRating'] = df_ratings.groupby('MovieID')['Rating'].transform(compute_rolling_averages)
    df_ratings = df_ratings.drop("Datetime", axis=1)

    df_all = df_ratings.merge(df_users, on="UserID").merge(df_movies, on="MovieID")

    return df_all



