from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

def load_llm():

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0
    )

    return llm


if __name__ == "__main__":

    llm = load_llm()

    response = llm.invoke("Who are you?")

    print(response.content)