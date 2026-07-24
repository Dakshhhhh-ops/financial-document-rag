from pathlib import Path
from langchain_community.document_loaders import PyMuPDFLoader


def load_documents():
    """
    Loads all PDF files from the data folder and
    returns a list of LangChain Document objects.
    """

    # Path to the data folder
    data_folder = Path("data")

    # Find all PDF files
    pdf_files = data_folder.glob("*.pdf")

    # Store all documents from all PDFs
    all_documents = []

    # Read every PDF
    for pdf_file in pdf_files:
        print(f"Loading: {pdf_file}")

        loader = PyMuPDFLoader(str(pdf_file))
        documents = loader.load()

        # Merge pages into one list
        all_documents.extend(documents)

    return all_documents


if __name__ == "__main__":
    docs = load_documents()

    print(f"\nTotal Documents: {len(docs)}")
    print(type(docs))
    print(type(docs[0]))
    print(docs[0])