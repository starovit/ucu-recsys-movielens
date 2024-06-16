import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score, roc_auc_score


# ML metrics
def ml_metrics(true_scores, predicted_scores, threshold=4):

    true_scores_cat = (true_scores >= threshold).astype(int)
    predicted_scores_cat = (predicted_scores >= threshold).astype(int)
    
    mae = mean_absolute_error(true_scores, predicted_scores)
    rmse = np.sqrt(mean_squared_error(true_scores, predicted_scores))

    precision = precision_score(true_scores_cat, predicted_scores_cat)
    recall = recall_score(true_scores_cat, predicted_scores_cat)
    f1 = f1_score(true_scores_cat, predicted_scores_cat)
    roc_auc = roc_auc_score(true_scores_cat, predicted_scores_cat)

    return {"mae": round(mae, 3),
            "rmse": round(rmse, 3),
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
            "roc_auc": round(roc_auc, 3)}



# Predictive metrics 
def precision_at_k(df, k=10, threshold=4):
    precisions = []
    for _, group in df.groupby('UserID'):
        if group.shape[0] >= k: # only if enough data for user
            sorted_group = group.sort_values(by='RatingPred', ascending=False)
            top_k = sorted_group.head(k)
            relevant_count = top_k[top_k['Rating'] >= threshold].shape[0]
            precision = relevant_count / k
            precisions.append(precision)
    
    return sum(precisions) / len(precisions) if precisions else 0,  len(precisions)

def recall_at_k(df, k=10, threshold=4):
    recalls = []
    for _, group in df.groupby('UserID'):
        relevant_items = group[group['Rating'] >= threshold]
        if len(relevant_items) == 0:  # only at least one good review
            continue
        if group.shape[0] >= k:  # only if enough data for user
            sorted_group = group.sort_values(by='RatingPred', ascending=False)
            top_k = sorted_group.head(k)
            relevant_in_top_k = top_k[top_k['Rating'] >= threshold].shape[0]
            recall = relevant_in_top_k / len(relevant_items)
            recalls.append(recall)

    return sum(recalls) / len(recalls) if recalls else 0

def average_precision_at_k(df, k=10, threshold=4):
    aps = []

    for _, group in df.groupby('UserID'):
        sorted_group = group.sort_values(by='RatingPred', ascending=False)
        top_k = sorted_group.head(k)

        relevant_items = group[group['Rating'] >= threshold]
        if len(relevant_items) == 0:
            continue

        relevant_in_top_k = top_k[top_k['Rating'] >= threshold]

        cumsum = 0
        precision_at_i = 0
        for i, item in enumerate(relevant_in_top_k.index, start=1):
            precision_at_i = len(relevant_in_top_k.iloc[:i]) / i
            if item in relevant_items.index:
                cumsum += precision_at_i
        
        ap = cumsum / min(len(relevant_items), k)
        aps.append(ap)

    return sum(aps) / len(aps) if aps else 0

def predictive_metrics(test: pd.DataFrame, predicted_scores, 
                       k=10, threshold=4):
    
    df = test.copy()
    df["RatingPred"] = predicted_scores
    prec_at_k, n_users_with_k = precision_at_k(df, k, threshold)
    rec_at_k = recall_at_k(df, k, threshold)
    avrg_prec_at_k = average_precision_at_k(df, k, threshold)

    return {"k": k,
            "threshold": threshold,
            "precision_at_k": round(prec_at_k, 3),
            "recall_at_k": round(rec_at_k, 3),
            "avrg_prec_at_k": round(avrg_prec_at_k, 3),
            "n_users_with_k": n_users_with_k
            }


# Rank metrics
def mean_reciprocal_rank(df, threshold=4):
    mrr = 0
    total_users = 0
    for _, group in df.groupby('UserID'):
        sorted_group = group.sort_values(by='RatingPred', ascending=False)
        sorted_group = sorted_group.reset_index()
        ranks = sorted_group[sorted_group['Rating'] >= threshold].index + 1
        if not ranks.empty:
            mrr += ranks[0]
            total_users += 1
    return mrr / total_users if total_users > 0 else 0

def hit_rate(df, k, rating_threshold=4, min_items_for_hit=1):
    hits = 0
    total_users = 0
    for _, group in df.groupby('UserID'):
        sorted_group = group.sort_values(by='RatingPred', ascending=False)
        top_k = sorted_group.head(k)
        if np.sum(top_k['Rating'] >= rating_threshold) >= min_items_for_hit:
            hits += 1
        total_users += 1

    return hits / total_users if total_users > 0 else 0


def rank_metrics(test: pd.DataFrame, predicted_scores,  k=10, threshold=4):
    
    df = test.copy()
    df["RatingPred"] = predicted_scores
    mean_r_r = mean_reciprocal_rank(df, threshold)
    hit_r = hit_rate(df, k, threshold=4)

    return {"mean_reciprocal_rank": round(mean_r_r, 3),
            "hit_rate": round(hit_r, 3)
            }

