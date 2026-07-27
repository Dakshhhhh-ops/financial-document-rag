from collections import defaultdict
from pathlib import Path

from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

from src.llm.llm import load_llm
from src.retrieval.retriever import get_retriever


def create_rag_chain():
    llm = load_llm()
    retriever = get_retriever()

    prompt = ChatPromptTemplate.from_template(
        """
        Answer the user's question only using the provided context.

        Context:
        {context}

        Question:
        {input}
        """
    )

    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    return retrieval_chain


#Ye function future me FastAPI aur Streamlit dono use karenge
def ask_question(question: str):

    chain = create_rag_chain()

    response = chain.invoke(
        {
            "input": question
        }
    )

    sources = defaultdict(set)

    for doc in response["context"]:
        file_name = Path(doc.metadata["source"]).name
        page = doc.metadata["page"] + 1

        sources[file_name].add(page)

    formatted_sources = []

    for file_name, pages in sources.items():
        formatted_sources.append(
            {
                "document": file_name,
                "pages": sorted(list(pages))
            }
        )

    return {
        "answer": response["answer"],
        "sources": formatted_sources
    }


if __name__ == "__main__":

    result = ask_question(
        "What was Apple's revenue in 2025?"
    )

    print("\nAnswer:\n")
    print(result["answer"])

    print("\nSources:\n")

    for source in result["sources"]:
        print(source["document"])
        print(f"Pages: {', '.join(map(str, source['pages']))}")
        print()