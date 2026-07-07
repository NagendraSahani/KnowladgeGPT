import ollama

from config import (
    OLLAMA_MODEL,
    OLLAMA_BASE_URL
)

client = ollama.Client(
    host = OLLAMA_BASE_URL
)


def ask_llm(prompt:str):
    response = client.chat(
        model=OLLAMA_MODEL,
        messages=[
            {
                "role":"user",
                "content": prompt
            }
        ]
    )


    return response["message"]["content"]
