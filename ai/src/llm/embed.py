import os
from langchain_openai.embeddings import OpenAIEmbeddings


openai_embeddings = OpenAIEmbeddings(
    model=os.getenv("OPENAI_EMBEDDING_MODEL_NAME"), api_key=os.getenv("OPENAI_API_KEY"), dimensions=768
)


def embed_text(text: str) -> list[float]:
    return openai_embeddings.embed_query(text)