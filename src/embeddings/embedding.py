from langchain_huggingface import HuggingFaceEmbeddings


def load_embedding_model():
    """
    Creates and returns the Hugging Face embedding model.
    """

    embedding_model = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5"
    )

    return embedding_model


if __name__ == "__main__":
    embedding_model = load_embedding_model()

    print(type(embedding_model))
    print(embedding_model)
