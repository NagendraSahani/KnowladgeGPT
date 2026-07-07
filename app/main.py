import streamlit as st
import tempfile
from loaders.pdf_loader import PDFLoader
from llm.ollama_client import ask_llm
from splitter.text_splitter import TextSplitter
from chains.rag_chain import RAGChain
from embeddings.embedding_model import EmbeddingModel
from vectorstore.chroma_db import ChromaVectorStore



st.set_page_config(
    page_title="KnowledgeGPT",
    page_icon="🤖",
    layout ="wide"

)

st.title("🤖 KnowledgeGPT")

uploaded_pdf = st.file_uploader(
    "Upload PDF",
    type=["pdf"]
)

if uploaded_pdf:
    with tempfile.NamedTemporaryFile(delete=False,suffix=".pdf") as temp_pdf:
        temp_pdf.write(uploaded_pdf.read())
        pdf_path = temp_pdf.name
    loader =PDFLoader()

    text = loader.load_pdf(pdf_path)
    st.success("PDF Loaded Successfully!")

    splitter = TextSplitter()

    chunks = splitter.split_text(text)
    st.success(f"Total Chunks: {len(chunks)}")
    st.text_area(
        "First Chunk",
        chunks[0],
        height=250
    )

    embedder = EmbeddingModel()

    with st.spinner("Generating Embeddings.."):
        embeddings = embedder.embed_documents(chunks)
    
    st.success("Embeddings Generated Succesfullly!")
    st.write(f"Total Embeddings : {len(embeddings)}")

    st.write(f"Embedding Dimension : {len(embeddings[0])}")

    vector_db = ChromaVectorStore()

    with st.spinner("Saving into ChromaDB..."):

        vector_db.add_documents(
            chunks,
            embeddings
        )

    st.success("Stored Successfully in ChromaDB!")

   
    st.write(f"Characters : {len(text)}")

    st.text_area(
        "preview",
        text[:3000],
        height=300

    )


question = st.text_area(
    "Ask Anything"

)


if st.button("Generate"):

    rag = RAGChain()

    with st.spinner("Searching Document..."):

        answer = rag.ask(question)

    st.success("Answer")

    st.write(answer)