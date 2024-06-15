import numpy as np
from numpy.linalg import norm


def cosine_similarity(vector1, vector2):
    cosine = np.dot(vector1, vector2) / (norm(vector2) * norm(vector2))
    return cosine
