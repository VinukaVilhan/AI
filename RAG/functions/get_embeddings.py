# from langchain_community.embeddings.bedrock import BedrockEmbeddings
from langchain_community.embeddings.ollama import OllamaEmbeddings

#key to the database
def get_embedding_function():
    embeddings = OllamaEmbeddings(model="llama3.1:latest")
    return embeddings

# def get_embedding_function():
#     embeddings = BedrockEmbeddings(
#         credentials_profile_name = "default", region_name="us-east-1"
#     )
#     return embeddings
