from sklearn.metrics.pairwise import cosine_similarity

def compare_features(f1,f2):

    score = cosine_similarity(
        f1.reshape(1,-1),
        f2.reshape(1,-1)
    )[0][0]

    return score