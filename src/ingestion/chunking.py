from loader import load_documents
from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_documents():
    """
    Splits loaded documents into smaller chunks.
    """

    documents = load_documents()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = text_splitter.split_documents(documents)

    return chunks


if __name__ == "__main__":
    chunks = split_documents()

    print(f"Total Chunks: {len(chunks)}")
    print(type(chunks[0]))
    print(chunks[0])