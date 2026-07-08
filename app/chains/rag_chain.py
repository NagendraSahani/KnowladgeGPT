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

    def detect_language_mode(self, question_lower):
        """
        Agar user Hindi/Hinglish me jawab maange (e.g. 'hindi me samjhao',
        'hinglish me batao', 'hindi mein explain karo'), to True return karo.
        """
        hindi_keywords = [
            "hindi me", "hindi mein", "hinglish me", "hinglish mein",
            "hindi m", "hinglish m", "hindi language", "in hindi",
            "samjhao hindi", "hindi samjhao", "hindi karo"
        ]
        return any(word in question_lower for word in hindi_keywords)

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

        use_hindi = self.detect_language_mode(question_lower)

        language_instruction = (
            "Respond in easy Hinglish (Hindi written in English/Roman script), "
            "mixed with simple English technical terms where needed. "
            "Explanation ekdum simple aur conversational tone me ho, jaise ek dost "
            "ya teacher samjhata hai."
            if use_hindi else
            "Respond in clear, simple English."
        )

        # ---------- Detailed Summary ----------
        if any(word in question_lower for word in [
            "summary", "summarize", "summarise",
            "notes", "overview"
        ]):
            prompt = f"""
You are an expert AI Study Assistant and teacher.
Your job is to read the given context (from a PDF) and generate an EXTREMELY DETAILED,
easy-to-understand summary/explanation — as if you are teaching this to a student from scratch.

Rules (follow strictly):
1. Cover EVERY important concept, definition, formula, and idea mentioned in the context — don't skip anything.
2. For EACH concept, do the following:
   - Explain it in simple, easy language (jaise koi teacher samjhata hai).
   - Give a real-life or practical EXAMPLE for that concept (agar context me example diya hai to wahi use karo, warna khud ek simple relevant example bana kar do).
   - If there's a formula/algorithm/process, break it down step-by-step.
3. Organize the summary using proper Headings and Sub-headings (##, ###).
4. Use bullet points and numbered lists wherever possible instead of long paragraphs.
5. Highlight important terms/keywords in **bold**.
6. Keep the tone simple and beginner-friendly, avoid unnecessary jargon.
7. Do NOT add any information that is not present in the context — only elaborate/explain what's already there, don't invent new facts.
8. At the end, add a short "Quick Recap" section with one-line bullet points summarizing everything.
9. {language_instruction}

Context:
{context}

Detailed Summary:
"""

        # ---------- Normal QA ----------
        else:
            prompt = f"""
You are an AI Study Assistant helping a student understand a document deeply.
Answer the question ONLY using the given context — do not use outside knowledge.

Rules:
1. First give a clear, direct answer to the question.
2. Then explain the concept behind the answer in simple language, as if teaching a beginner.
3. Give a relevant EXAMPLE to make the concept easier to understand (use example from context if available, else create a simple one based only on context info).
4. Use bullet points / step-by-step breakdown if the answer involves a process, formula, or multiple points.
5. If the answer is not present in the context, reply exactly:
"I couldn't find this information in the uploaded document."
6. Do not make up information that isn't in the context.
7. {language_instruction}

Context:
{context}

Question:
{question}

Detailed Answer:
"""

        answer = ask_llm(prompt)
        return {
            "answer": answer,
            "sources": retrieved_docs
        }
