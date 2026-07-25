from langchain_community.vectorstores import FAISS

from src.embeddings.embedding import load_embedding_model


def load_vector_store():
    """
    Loads the saved FAISS vector database.
    """

    embedding_model = load_embedding_model()

    vector_store = FAISS.load_local(
        "faiss_index",
        embedding_model,
        allow_dangerous_deserialization=True
    )

    return vector_store

def get_retriever():
    """
    Returns a LangChain Retriever.
    """

    vector_store = load_vector_store()

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )

    return retriever

if __name__ == "__main__":

    retriever = get_retriever()

    query = "What was Apple's revenue in 2025?"

    documents = retriever.invoke(query)

    print(f"Retrieved {len(documents)} documents\n")

    for i, doc in enumerate(documents, 1):
        print(f"Chunk {i}")
        print("-" * 50)
        print(doc.page_content[:400])
        print()
