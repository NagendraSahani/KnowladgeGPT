from services.document_service import DocumentService
from llm.ollama_client import ask_llm


class SummaryChain:

    def summarize(
        self,
        language="English",
        summary_type="Detailed"
    ):

        document = DocumentService.get_document()

        if not document.strip():
            return "❌ No document found. Please upload a PDF first."

        prompt = f"""
You are KnowledgeGPT.

You are an expert AI Study Assistant.

Your task is to generate a {summary_type} summary from ONLY the uploaded document.

==============================
LANGUAGE RULES
==============================

Language Selected: {language}

1. If English is selected:
- Reply ONLY in English.

2. If Hindi is selected:
- Reply ONLY in Hindi.
- Use proper Devanagari script.

3. If Hinglish is selected:
- Reply ONLY in Hinglish.
- Hindi words should be written using English letters.
- DO NOT use Hindi (Devanagari) script.
- DO NOT reply completely in English.

==============================
SUMMARY RULES
==============================

• Do NOT hallucinate.
• Use ONLY information available in the document.
• Do NOT add outside knowledge.
• Explain in simple language.
• Use proper headings.
• Use bullet points.
• Highlight important concepts.
• Mention formulas if available.
• Mention important definitions.
• Mention applications if present.

==============================
SUMMARY STYLE
==============================

If Summary Type = Short
→ 5-10 important points only.

If Summary Type = Detailed
→ Explain every major topic with headings.

If Summary Type = Bullet
→ Only bullet points.

If Summary Type = Exam
→ Focus on important exam topics.

==============================
DOCUMENT
==============================

{document}

==============================
OUTPUT
==============================

Generate the summary now.
"""

        answer = ask_llm(prompt)

        return answer