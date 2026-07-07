from dotenv import load_dotenv
import os 

load_dotenv()

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")
OLLAMA_BASE_URL =os.getenv("OLLAMA_BASE_URL")

print(OLLAMA_MODEL,OLLAMA_BASE_URL)