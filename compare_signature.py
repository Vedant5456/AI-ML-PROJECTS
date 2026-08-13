from sklearn.metrics.pairwise import cosine_similarity

def compare(f1, f2):

    similarity = cosine_similarity([f1],[f2])[0][0]

    return similarity