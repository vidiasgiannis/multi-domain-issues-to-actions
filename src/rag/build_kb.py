import faiss
class KBIndex:
    def __init__(self, embedder, texts, ids):
        self.embedder = embedder
        self.ids = ids
        emb = embedder.encode(texts)
        self.index = faiss.IndexFlatIP(emb.shape[1])
        self.index.add(emb)
    def search(self, query, k=2):
        q = self.embedder.encode([query])
        D,I = self.index.search(q, k)
        return I[0], D[0]
