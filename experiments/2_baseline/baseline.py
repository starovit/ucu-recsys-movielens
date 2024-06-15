def recommend_by_rating_sum(ratings, n_items=10) -> np.ndarray:
    rating_sum_df = ratings.groupby('MovieID').agg(
        RatingSum=('Rating', sum)
    ).reset_index()
    n_ids = rating_sum_df.sort_values(by='RatingSum', ascending=False).reset_index(drop=True).loc[:n_items, 'MovieID']
    return n_ids


def recommend_rating_avg(ratings, n_items=10) -> np.ndarray:
    rating_avg_df = ratings.groupby('MovieID').agg(
        RatingSum=('Rating', sum),
        UserAmount=('UserID', len)
    ).reset_index()
    n_ids = rating_avg_df.sort_values(by='AvgRating', ascending=False).reset_index(drop=True).loc[:n_items, 'MovieID']
    return n_ids


def recommend_interaction_count(ratings, n_items=10) -> np.ndarray:
    interaction_count_df = ratings.groupby('MovieID').agg(
        UserAmount=('UserID', len)
    ).reset_index()
    n_ids = interaction_count_df.sort_values(by='UserAmount', ascending=False).reset_index(drop=True).loc[:n_items, 'MovieID']
    return n_ids