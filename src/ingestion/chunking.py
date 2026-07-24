from langchain_text_splitters import RecursiveCharacterTextSplitter
from loader import load_documents
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)