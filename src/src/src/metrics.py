from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import precision_score, recall_score

def get_ml_metrics(true_scores, predict_scores, threshold=4):

    true_scores_cat = (true_scores >= threshold).astype(int)
    predict_scores_cat = (predict_scores >= threshold).astype(int)
    
    mae = mean_absolute_error(true_scores, predict_scores)
    rmse = np.sqrt(mean_squared_error(true_scores, predict_scores))
    precision = precision_score(true_scores_cat, predict_scores_cat)
    recall = recall_score(true_scores_cat, predict_scores_cat)

    return {"mae": round(mae, 3),
            "rmse": round(rmse, 3),
            "precision": round(precision, 3),
            "recall": round(recall, 3)}