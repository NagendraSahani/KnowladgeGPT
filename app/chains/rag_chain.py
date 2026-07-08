from retriever.retriever import Retriever
from llm.llm_factory import ask_llm


class RAGChain:

    def __init__(self):
        self.retriever = Retriever()

    def build_context(self, docs):
        context = ""
        for doc in docs:
            context += doc["text"] + "\n\n"

        return context

    def ask(self, question):

        retrieved_docs = self.retriever.retrieve(question)

        # Agar koi relevant chunk mila hi nahi, to LLM call kiye bina
        # seedha graceful message return karo.
        if not retrieved_docs:
            return {
                "answer": "I couldn't find this information in the uploaded document.",
                "sources": []
            }

        context = self.build_context(retrieved_docs)
        question_lower = question.lower()

        # ---------- Summary ----------
        if any(word in question_lower for word in [
            "summary", "summarize", "summarise",
            "notes", "overview"
        ]):

            prompt = f"""
You are an AI Study Assistant.

Read the following context and generate a detailed summary.

Rules:
- Cover all important concepts.
- Use headings.
- Use bullet points wherever possible.
- Keep the explanation easy to understand.
- Do not add information outside the context.

Context:

{context}

Summary:
"""

        # ---------- Normal QA ----------
        else:

            prompt = f"""
You are an AI assistant.

Answer ONLY from the given context.

If the answer is not present, reply:

"I couldn't find this information in the uploaded document."

Context:

{context}

Question:

{question}

Answer:
"""

        answer = ask_llm(prompt)

        return {
            "answer": answer,
            "sources": retrieved_docs
        }