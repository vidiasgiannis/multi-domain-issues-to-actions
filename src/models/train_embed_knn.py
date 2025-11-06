from sklearn.neighbors import KNeighborsClassifier
def train_embed_knn(emb_src, y_src, k=3):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(emb_src, y_src)
    return knn
