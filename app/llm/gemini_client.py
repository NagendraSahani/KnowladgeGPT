from google import genai
from google.genai import types

from config import (
    GEMINI_API_KEY,
    GEMINI_MODEL
)

# Create Gemini Client
client = genai.Client(
    api_key=GEMINI_API_KEY
)


def ask_llm(prompt: str) -> str:
    """
    Send prompt to Gemini and return response text.
    """

    try:

        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.3,
                max_output_tokens=2048,
            ),
        )

        return response.text

    except Exception as e:

        return f"❌ Gemini Error:\n{str(e)}"