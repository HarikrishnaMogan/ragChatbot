"""
RAG Chatbot with PDF/Word upload, ChromaDB, HuggingFace all-MiniLM-L6-v2 embeddings,
LangChain, and Streamlit UI.

Compatible with Python 3.12
Run with:
    streamlit run app.py
"""

import os
import tempfile
import shutil

import streamlit as st

import pypdf
import docx2txt
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "rag_docs"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

st.set_page_config(page_title="RAG Chatbot", page_icon="📄", layout="wide")


# --------------------------------------------------------------------------- #
# Cached resources
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner=False)
def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)


def load_document(file_path: str, file_name: str):
    """Load a PDF or Word document into LangChain Document objects."""
    ext = os.path.splitext(file_name)[1].lower()

    if ext == ".pdf":
        reader = pypdf.PdfReader(file_path)
        docs = []
        for page_num, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                docs.append(Document(page_content=text, metadata={"page": page_num}))
        return docs

    elif ext in (".docx", ".doc"):
        text = docx2txt.process(file_path)
        if text and text.strip():
            return [Document(page_content=text, metadata={})]
        return []

    else:
        raise ValueError(f"Unsupported file type: {ext}")


def build_vectorstore(uploaded_files, embeddings):
    """Save uploads to disk, split into chunks, and index them in Chroma."""
    all_chunks = []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    tmp_dir = tempfile.mkdtemp()
    try:
        for uploaded_file in uploaded_files:
            tmp_path = os.path.join(tmp_dir, uploaded_file.name)
            with open(tmp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            docs = load_document(tmp_path, uploaded_file.name)
            for d in docs:
                d.metadata["source"] = uploaded_file.name

            chunks = splitter.split_documents(docs)
            all_chunks.extend(chunks)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    if not all_chunks:
        return None

    # Fresh collection each time documents are (re)indexed
    if os.path.exists(PERSIST_DIR):
        shutil.rmtree(PERSIST_DIR, ignore_errors=True)

    vectorstore = Chroma.from_documents(
        documents=all_chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=PERSIST_DIR,
    )
    return vectorstore


def get_llm(groq_api_key, model_name="llama-3.1-8b-instant"):
    """Initialize a Groq chat model via LangChain's init_chat_model."""
    return init_chat_model(
        model_name,
        model_provider="groq",
        api_key=groq_api_key,
        temperature=0,
    )


def get_prompt():
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant answering questions using only the "
                "provided context from the user's uploaded documents. "
                "If the answer isn't in the context, say you don't know.\n\n"
                "Context:\n{context}",
            ),
            ("human", "{question}"),
        ]
    )


def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)


def get_rag_chain(vectorstore, groq_api_key, model_name="llama-3.1-8b-instant"):
    llm = get_llm(groq_api_key, model_name)
    prompt = get_prompt()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain, retriever


# --------------------------------------------------------------------------- #
# Streamlit UI
# --------------------------------------------------------------------------- #
def main():
    st.title("📄 RAG Chatbot — PDF & Word")
    st.caption("ChromaDB + HuggingFace all-MiniLM-L6-v2 embeddings + LangChain")

    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    with st.sidebar:
        st.header("Setup")

        groq_api_key = st.text_input(
            "Groq API key (for the chat LLM)",
            type="password",
            help="Used only to generate answers. Embeddings run locally via HuggingFace.",
        )
        groq_model = st.selectbox(
            "Groq model",
            ["openai/gpt-oss-120b", "qwen/qwen3.8-27b"],
            index=0,
        )

        st.divider()
        st.subheader("Upload documents")
        uploaded_files = st.file_uploader(
            "Upload PDF or Word files",
            type=["pdf", "docx", "doc"],
            accept_multiple_files=True,
        )

        if st.button("Build / Rebuild Knowledge Base", disabled=not uploaded_files):
            with st.spinner("Embedding and indexing documents..."):
                embeddings = get_embeddings()
                vectorstore = build_vectorstore(uploaded_files, embeddings)
                if vectorstore is None:
                    st.error("No text could be extracted from the uploaded files.")
                else:
                    st.session_state.vectorstore = vectorstore
                    st.session_state.chat_history = []
                    st.success(f"Indexed {len(uploaded_files)} file(s) successfully!")

        if st.session_state.vectorstore is not None:
            st.success("✅ Knowledge base ready")
        else:
            st.info("Upload files and click the button to get started.")

        st.divider()
        if st.button("Clear chat"):
            st.session_state.chat_history = []

    # --- Chat area ---
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("Sources"):
                    for s in msg["sources"]:
                        st.markdown(f"- **{s}**")

    user_question = st.chat_input("Ask a question about your documents...")

    if user_question:
        if st.session_state.vectorstore is None:
            st.warning("Please upload and index documents first.")
            return
        if not groq_api_key:
            st.warning("Please enter your Groq API key in the sidebar.")
            return

        st.session_state.chat_history.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                chain, retriever = get_rag_chain(
                    st.session_state.vectorstore, groq_api_key, groq_model
                )
                source_docs = retriever.invoke(user_question)
                answer = chain.invoke(user_question)
                sources = sorted(
                    {doc.metadata.get("source", "unknown") for doc in source_docs}
                )
                st.markdown(answer)
                if sources:
                    with st.expander("Sources"):
                        for s in sources:
                            st.markdown(f"- **{s}**")

        st.session_state.chat_history.append(
            {"role": "assistant", "content": answer, "sources": sources}
        )


if __name__ == "__main__":
    main()