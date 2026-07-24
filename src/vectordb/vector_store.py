from langchain_community.vectorstores import FAISS

from src.ingestion.chunking import split_documents
from src.embeddings.embedding import load_embedding_model

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
    vector_store.save_local("faiss_index")
    return vector_store

if __name__ == "__main__":
    print("Creating FAISS vector store...")

    vector_store = create_vector_store()

    print("Vector store created successfully!")