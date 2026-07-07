from pathlib import Path
import uuid
import chromadb


class ChromaVectorStore:

    def __init__(self):

        # Project Root
        BASE_DIR = Path(__file__).resolve().parents[2]

        # Database Folder
        DB_PATH = BASE_DIR / "chroma_db"

        self.client = chromadb.PersistentClient(
            path=str(DB_PATH)
        )

        self.collection = self.client.get_or_create_collection(
            name="knowledgegpt"
        )

    def add_documents(self, chunks, embeddings):

        ids = []
        metadatas = []

        for i, chunk in enumerate(chunks):

            ids.append(str(uuid.uuid4()))

            metadatas.append({
                "chunk": i
            })

        self.collection.add(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas
        )

    def similarity_search(self, query_embedding, top_k=5):

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=[
                "documents",
                "metadatas",
                "distances"
            ]
        )

        return results

    def count_documents(self):

        return self.collection.count()

    def delete_collection(self):

        self.client.delete_collection(
            "knowledgegpt"
        )

        self.collection = self.client.get_or_create_collection(
            name="knowledgegpt"
        )