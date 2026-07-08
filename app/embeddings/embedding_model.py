from sentence_transformers import SentenceTransformer


class EmbeddingModel:

    def __init__(self):
        self.model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )

    def embed_documents(self, documents):

        return self.model.encode(
            documents,
            convert_to_numpy=True
        ).tolist()

    def embed_query(self, query):

        return self.model.encode(
            query,
            convert_to_numpy=True
        ).tolist()
