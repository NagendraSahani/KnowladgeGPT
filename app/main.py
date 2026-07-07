import os
import tempfile

import streamlit as st

from loaders.pdf_loader import PDFLoader
from splitter.text_splitter import TextSplitter
from chains.rag_chain import RAGChain
from chains.summary_chain import SummaryChain
from embeddings.embedding_model import EmbeddingModel
from vectorstore.chroma_db import ChromaVectorStore
from memory.chat_memory import ChatMemory
from services.document_service import DocumentService

# ==========================
# INIT
# ==========================
ChatMemory.initialize()

st.set_page_config(
    page_title="KnowledgeGPT",
    page_icon="🤖",
    layout="wide",
)

if "processed_pdf_id" not in st.session_state:
    st.session_state.processed_pdf_id = None


# ==========================
# SIDEBAR (single source of truth — no duplicate widgets)
# ==========================
with st.sidebar:
    st.title("🤖 KnowledgeGPT")

    mode = st.selectbox(
        "Mode",
        ["Chat", "Summary"],
    )

    language = st.selectbox(
        "Language",
        ["English", "Hindi", "Hinglish"],
    )

    summary_type = st.selectbox(
        "Summary Type",
        ["Short", "Detailed", "Bullet"],
    )

    st.markdown("---")

    if st.button("🗑 Clear Chat"):
        ChatMemory.clear()
        st.rerun()


# ==========================
# MAIN AREA
# ==========================
st.title("🤖 KnowledgeGPT")

uploaded_pdf = st.file_uploader("Upload PDF", type=["pdf"])

if uploaded_pdf:
    # Unique identity for this upload (name + size). Used so we don't
    # re-embed / re-insert into Chroma on every Streamlit rerun.
    pdf_identity = f"{uploaded_pdf.name}_{uploaded_pdf.size}"

    if st.session_state.processed_pdf_id != pdf_identity:
        pdf_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
                temp_pdf.write(uploaded_pdf.read())
                pdf_path = temp_pdf.name

            loader = PDFLoader()
            text = loader.load_pdf(pdf_path)
            DocumentService.save_document(text)

            st.success("PDF Loaded Successfully!")

            splitter = TextSplitter()
            chunks = splitter.split_text(text)
            st.success(f"Total Chunks: {len(chunks)}")

            embedder = EmbeddingModel()
            with st.spinner("Generating Embeddings.."):
                embeddings = embedder.embed_documents(chunks)

            st.success("Embeddings Generated Successfully!")
            st.write(f"Total Embeddings : {len(embeddings)}")
            st.write(f"Embedding Dimension : {len(embeddings[0])}")

            vector_db = ChromaVectorStore()
            with st.spinner("Saving into ChromaDB..."):
                vector_db.add_documents(chunks, embeddings)

            st.success("Stored Successfully in ChromaDB!")

            # Cache everything needed later in this session
            st.session_state.processed_pdf_id = pdf_identity
            st.session_state.pdf_text = text
            st.session_state.pdf_chunks = chunks

        finally:
            # Clean up temp file regardless of success/failure
            if pdf_path and os.path.exists(pdf_path):
                os.remove(pdf_path)
    else:
        st.info("This PDF is already processed and stored in ChromaDB.")

    # Show cached preview info without re-processing
    text = st.session_state.get("pdf_text", "")
    chunks = st.session_state.get("pdf_chunks", [])

    if chunks:
        st.text_area("First Chunk", chunks[0], height=250)
    if text:
        st.write(f"Characters : {len(text)}")
        st.text_area("Preview", text[:3000], height=300)


# ==========================
# CHAT HISTORY
# ==========================
for message in ChatMemory.get():
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ==========================
# QUESTION INPUT
# ==========================
question = st.text_area("Ask Anything")


# ==========================
# GENERATE BUTTON
# ==========================
if st.button("Generate"):

    if uploaded_pdf is None:
        st.warning("⚠ Please upload a PDF first.")

    elif mode == "Chat":
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            rag = RAGChain()

            with st.spinner("Searching document..."):
                result = rag.ask(question)

            ChatMemory.add("user", question)
            ChatMemory.add("assistant", result["answer"])

            st.markdown("## 🤖 Answer")
            st.write(result["answer"])

            st.markdown("---")
            st.markdown("## 📄 Sources")

            for source in result["sources"]:
                st.write(f"📌 Chunk : {source['metadata']['chunk']}")
                st.write(f"📏 Similarity : {source['distance']:.4f}")
                st.write(source["text"][:250] + "...")
                st.markdown("---")

    elif mode == "Summary":
        with st.spinner("Generating Summary..."):
            summary = SummaryChain().summarize(
                language=language,
                summary_type=summary_type,
            )

        st.markdown("## 📝 Summary")
        st.write(summary)