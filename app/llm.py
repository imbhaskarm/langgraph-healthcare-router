import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()


def get_llm() -> ChatGroq:
    """
    Return a ChatGroq instance using the GROQ_API_KEY from .env.

    Using llama-3.3-70b-versatile because it reliably follows
    structured_output constraints for both category and sentiment classification.
    """
    # ⚠️ Updated from deprecated OpenAI/GPT-4 usage → Groq via langchain-groq (latest as of 2025)
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=os.getenv("GROQ_API_KEY"),
    )
