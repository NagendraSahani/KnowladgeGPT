from embeddings.embedding_model import EmbeddingModel
from vectorstore.chroma_db import ChromaVectorStore


class Retriever:

    def __init__(self):
        self.embedding_model = EmbeddingModel()
        self.vector_db = ChromaVectorStore()

    def retrieve(self, query, top_k=5):

        query_embedding = self.embedding_model.embed_query(query)

        results = self.vector_db.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=[
                "documents",
                "metadatas",
                "distances"
            ]
        )

        retrieved_docs = []

        docs = results["documents"][0]
        metas = results["metadatas"][0]
        distances = results["distances"][0]

        for doc, meta, distance in zip(docs, metas, distances):
            retrieved_docs.append({
                "text": doc,
                "metadata": meta,
                "distance": distance
            })

        return retrieved_docs