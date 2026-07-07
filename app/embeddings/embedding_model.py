import ollama 
from config import OLLAMA_BASE_URL

class EmbeddingModel:
    def __init__(self):
        self.client=ollama.Client(host=OLLAMA_BASE_URL)
        self.model="nomic-embed-text:latest"

    def embed_documents(self,texts):
        embeddings = []

        for text in texts:
            response = self.client.embed(
                model=self.model,
                input=text
            )
            embeddings.append(response["embeddings"][0])
        return embeddings

    def embed_query(self,query):
        response =self.client.embed(
            model = self.model,
            input=query
        )

        return response["embeddings"][0]