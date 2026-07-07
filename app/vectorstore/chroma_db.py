import chromadb


class ChromaVectorStore:

    def __init__(self):

        self.client = chromadb.PersistentClient(
            path="../chroma_db"
        )

        self.collection = self.client.get_or_create_collection(
            name="knowledgegpt"
        )

    def add_documents(self, chunks, embeddings):

        ids = []

        for i in range(len(chunks)):
            ids.append(f"chunk_{i}")

        self.collection.add(
            ids=ids,
            documents=chunks,
            embeddings=embeddings
        )

    def similarity_search(self, embedding, top_k=5):

        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
            include=["documents"]
        )
        return results["documents"][0]