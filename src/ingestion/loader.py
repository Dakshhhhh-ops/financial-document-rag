from pathlib import Path
from langchain_community.document_loaders import PyMuPDFLoader

data_folder=Path("data")
pdf_files=data_folder.glob("*.pdf") #returns a generator 
all_documents=[]
for pdf_file in pdf_files:
    print(pdf_file)
    loader = PyMuPDFLoader(str(pdf_file))
    documents=loader.load()
    # Add all pages from this PDF to the main list
    all_documents.extend(documents)
print(len(all_documents))
print(type(all_documents[0]))
print(all_documents[0])
    
