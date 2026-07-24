from langchain_community.vectorstores import FAISS

from ingestion.chunking import split_documents
from embeddings.embedding import load_embedding_model

def create_vector_store():
    """
    Creates and returns the FAISS vector store
    """
    chunks=split_documents()
    embedding_model=load_embedding_model()
    vector_store=FAISS.from_documents(
        documents=chunks,
        embedding=embedding_model
    )
    return vector_store