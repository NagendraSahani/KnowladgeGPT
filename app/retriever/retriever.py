from embeddings.embedding_model import EmbeddingModel
from vectorstore.chroma_db import ChromaVectorStore


class Retriever:

    def __init__(self):
        self.embedder = EmbeddingModel()
        self.vector_db = ChromaVectorStore()

    def retrieve(self, query, top_k=5):

        query_embedding = self.embedder.embed_query(query)

        results = self.vector_db.similarity_search(
            query_embedding,
            top_k
        )

        return results