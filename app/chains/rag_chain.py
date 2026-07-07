from retriever.retriever import Retriever
from llm.ollama_client import ask_llm


class RAGChain:

    def __init__(self):

        self.retriever = Retriever()

    def ask(self, question):

        documents = self.retriever.retrieve(question)

        context = "\n\n".join(documents)

        prompt = f"""
You are an AI assistant.

Answer ONLY from the given context.

If the answer is not present,
reply:
"I couldn't find this information in the uploaded document."

Context:

{context}

Question:

{question}

Answer:
"""

        return ask_llm(prompt)