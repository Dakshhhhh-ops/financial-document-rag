from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from src.llm.llm import load_llm
from src.retrieval.retriever import get_retriever

def create_rag_chain():
    llm=load_llm()
    retriever=get_retriever()
    prompt = ChatPromptTemplate.from_template(
        """
        Answer the user's question only using the provided context.

        Context:
        {context}

        Question:
        {input}
        """
    )
    document_chain=create_stuff_documents_chain(llm,prompt)
    retrieval_chain=create_retrieval_chain(retriever,document_chain)
    return retrieval_chain

if __name__ == "__main__":

    chain = create_rag_chain()

    response = chain.invoke(
        {"input": "What was Apple's revenue in 2025?"}
    )

    print(response["answer"])