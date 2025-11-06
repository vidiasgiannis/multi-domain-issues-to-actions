import os
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv()


def get_azure_client():
    """Initialize and return Azure OpenAI client."""
    return AzureOpenAI(
        api_version = os.getenv("OPENAI_API_VERSION"),
        api_key = os.getenv("OPENAI_API_KEY"),
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"),
    )

def embed_text(texts, batch_size=8):
    """Generate embeddings for a list of texts using Azure OpenAI."""
    client = get_azure_client()
    embeddings = []
    model = os.getenv("AZURE_OPENAI_EMB_API_DEPLOYMENT")
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        response = client.embeddings.create(
            input=batch,
            model=model
        )
        batch_embeddings = [data.embedding for data in response.data]
        embeddings.extend(batch_embeddings)
    return embeddings