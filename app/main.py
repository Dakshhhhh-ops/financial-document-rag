import time

import requests
import streamlit as st

# ==========================
# Configuration
# ==========================

API_URL = "http://127.0.0.1:8000/ask"
HEALTH_URL = API_URL.rsplit("/", 1)[0] + "/health"  # best-effort health probe

MODEL_NAME = "Llama 3.3 70B"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
VECTOR_STORE = "FAISS"
BACKEND_NAME = "FastAPI"

TECH_STACK = ["LangChain", "FAISS", "HuggingFace", "Groq", "FastAPI", "Streamlit"]

SUGGESTED_QUESTIONS = [
    "What was Apple's revenue in 2025?",
    "Compare Apple and NVIDIA revenue.",
    "What are Apple's biggest risks?",
    "Summarize Amazon's annual report.",
]

st.set_page_config(
    page_title="Financial Document RAG",
    page_icon="📄",
    layout="wide",
)

# ==========================
# Session State
# ==========================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending_question" not in st.session_state:
    st.session_state.pending_question = None


# ==========================
# Helper Functions
# ==========================

def call_api(question: str) -> dict:
    """Send the user's question to the FastAPI backend and return the JSON response."""
    response = requests.post(
        API_URL,
        json={"question": question},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def check_backend_status() -> bool:
    """Best-effort check to see if the backend is reachable. Called on every rerun,
    so the sidebar status reflects the backend's current state each time the user
    sends a prompt or interacts with the app."""
    try:
        requests.get(HEALTH_URL, timeout=3)
        return True
    except requests.exceptions.RequestException:
        # Fall back to the base API URL in case /health doesn't exist
        try:
            requests.get(API_URL.rsplit("/", 1)[0], timeout=3)
            return True
        except requests.exceptions.RequestException:
            return False


def clear_chat() -> None:
    """Reset the conversation history."""
    st.session_state.messages = []
    st.session_state.pending_question = None


def build_transcript() -> str:
    """Build a plain-text/markdown transcript of the conversation for download."""
    lines = ["# Financial Document RAG — Conversation\n"]
    for message in st.session_state.messages:
        speaker = "You" if message["role"] == "user" else "Assistant"
        lines.append(f"**{speaker}** ({message.get('timestamp', '')}):\n{message['content']}\n")
    return "\n".join(lines)


def display_sources(sources: list) -> None:
    """Render document citations as expandable cards, one per document."""
    if not sources:
        return

    st.markdown(f"**📚 Sources ({len(sources)})**")

    for source in sources:
        document = source.get("document", "Unknown Document")
        pages = source.get("pages", [])

        with st.expander(f"📄 {document}"):
            if pages:
                st.markdown("**Pages:**")
                for page in pages:
                    st.markdown(f"- Page {page}")
            else:
                st.caption("No page information available.")


def display_answer_metadata(sources: list, elapsed: float) -> None:
    """Show a compact, low-key stats line: docs retrieved and response time."""
    st.caption(f"📄 Retrieved Docs: {len(sources)}  &nbsp;|&nbsp;  ⏱ Response Time: {elapsed:.2f}s")


def render_message(message: dict) -> None:
    """Render a single chat message, including sources, metadata, and timestamp."""
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant":
            if "elapsed" in message:
                display_answer_metadata(message.get("sources", []), message["elapsed"])

            display_sources(message.get("sources", []))

            if st.button("📋 Copy Answer", key=f"copy_{id(message)}"):
                st.code(message["content"], language=None)

        if message.get("timestamp"):
            st.caption(message["timestamp"])


def handle_question(question: str) -> None:
    """Send a question to the backend, stream progress, and store the result."""
    question = question.strip()
    if not question:
        return

    now = time.strftime("%I:%M %p")
    st.session_state.messages.append({"role": "user", "content": question, "timestamp": now})

    with st.chat_message("user"):
        st.markdown(question)
        st.caption(now)

    with st.chat_message("assistant"):
        status_box = st.empty()
        start_time = time.time()

        try:
            status_box.info("🔍 Searching vector store...")
            time.sleep(0.3)
            status_box.info("📚 Retrieving documents...")
            time.sleep(0.3)
            status_box.info("🤖 Generating answer...")

            result = call_api(question)
            elapsed = time.time() - start_time
            status_box.empty()

            answer = result.get("answer", "No answer returned.")
            sources = result.get("sources", [])
            answer_time = time.strftime("%I:%M %p")

            st.markdown(answer)
            display_answer_metadata(sources, elapsed)
            display_sources(sources)
            st.caption(answer_time)

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                    "elapsed": elapsed,
                    "timestamp": answer_time,
                }
            )

        except requests.exceptions.ConnectionError:
            status_box.empty()
            st.error("❌ Unable to connect to the FastAPI server. Please make sure it's running.")

        except requests.exceptions.Timeout:
            status_box.empty()
            st.error("⏳ The request timed out. Please try again.")

        except requests.exceptions.HTTPError as e:
            status_box.empty()
            st.error(f"⚠️ HTTP Error from backend: {e}")

        except Exception as e:
            status_box.empty()
            st.error(f"🚨 Unexpected Error: {e}")


# ==========================
# Sidebar
# ==========================

with st.sidebar:
    st.markdown("### 📄 Financial Document RAG")
    st.divider()

    col1, col2 = st.columns(2)
    if col1.button("🆕 New Chat", use_container_width=True):
        clear_chat()
        st.rerun()

    if col2.button("🗑 Clear", use_container_width=True):
        clear_chat()
        st.rerun()

    if st.session_state.messages:
        st.download_button(
            "⬇ Download Conversation",
            data=build_transcript(),
            file_name="conversation.md",
            mime="text/markdown",
            use_container_width=True,
        )

    st.divider()

    with st.expander("⚙ System Information"):
        st.markdown("**Model**")
        st.caption(MODEL_NAME)

        st.markdown("**Embeddings**")
        st.caption(EMBEDDING_MODEL)

        st.markdown("**Vector Store**")
        st.caption(VECTOR_STORE)

        st.markdown("**Backend**")
        st.caption(BACKEND_NAME)

    st.divider()

    st.markdown("**Application Status**")
    if check_backend_status():
        st.success("🟢 API Connected")
    else:
        st.error("🔴 API Offline")

    st.divider()

    st.markdown("**About**")
    st.caption("Financial Document RAG — built with:")
    st.caption(" • " + "\n • ".join(TECH_STACK))

# ==========================
# Header
# ==========================

st.title("📄 Financial Document RAG")
st.caption("Ask questions about financial reports.")
st.divider()

# ==========================
# Empty State — Suggested Questions
# ==========================

if not st.session_state.messages:
    st.markdown("💡 **Try asking:**")
    cols = st.columns(2)
    for i, question in enumerate(SUGGESTED_QUESTIONS):
        if cols[i % 2].button(question, use_container_width=True):
            st.session_state.pending_question = question
            st.rerun()

# ==========================
# Display Chat History
# ==========================

for message in st.session_state.messages:
    render_message(message)

# ==========================
# Chat Input
# ==========================

user_question = st.chat_input("Ask a question about financial reports...")

# A suggested question click takes priority if the user didn't type one directly
if st.session_state.pending_question and not user_question:
    user_question = st.session_state.pending_question
    st.session_state.pending_question = None

if user_question and user_question.strip():
    handle_question(user_question)

# ==========================
# Footer
# ==========================

st.divider()
st.caption("Powered by Llama 3.3 • FAISS • HuggingFace BGE • FastAPI • Streamlit")