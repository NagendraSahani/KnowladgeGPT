from config import LLM_PROVIDER

if LLM_PROVIDER.lower() == "gemini":
    from llm.gemini_client import ask_llm

elif LLM_PROVIDER.lower() == "ollama":
    from llm.ollama_client import ask_llm

else:
    raise ValueError(f"Unsupported LLM Provider: {LLM_PROVIDER}")