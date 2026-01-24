import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.output_parsers import PydanticOutputParser
from sklearn.metrics.pairwise import cosine_similarity
from pydantic import BaseModel, Field
from typing import List
from pypdf import PdfReader
import pptx
import os

# ----------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------
st.set_page_config(
    page_title="AI Document Search",
    page_icon="📘",
    layout="wide"
)

# ----------------------------------------------------
# STYLES
# ----------------------------------------------------
st.markdown("""
<style>
.chat-box {
    background-color: #f8f9fa;
    padding: 12px;
    border-radius: 10px;
    margin-bottom: 10px;
}
.user-msg {
    background-color: #dcf8c6;
    padding: 10px;
    border-radius: 8px;
}
.bot-msg {
    background-color: #eeeeee;
    padding: 10px;
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# LOAD ENV
# ----------------------------------------------------
load_dotenv()

# ----------------------------------------------------
# DATA MODEL
# ----------------------------------------------------
class RAGAnswer(BaseModel):
    answer: str = Field(description="Answer to the question")
    confidence: str = Field(description="high / medium / low")
    source_chunks: List[str] = Field(description="Top relevant chunks")

# ----------------------------------------------------
# LOAD MODELS
# ----------------------------------------------------
@st.cache_resource
def load_models():
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
    parser = PydanticOutputParser(pydantic_object=RAGAnswer)
    return llm, embeddings, parser

llm, embeddings, parser = load_models()

# ----------------------------------------------------
# FUNCTIONS
# ----------------------------------------------------
def process_documents(text):
    return [p.strip() for p in text.split("\n\n") if p.strip()]

def rag_query(chunks, query, top_k=3):
    doc_embeddings = embeddings.embed_documents(chunks)
    query_embedding = embeddings.embed_query(query)

    scores = cosine_similarity([query_embedding], doc_embeddings)[0]
    top_indices = scores.argsort()[-top_k:][::-1]

    sources = [chunks[i] for i in top_indices]
    context = "\n\n".join(sources)

    prompt = f"""
Answer ONLY from the context.

CONTEXT:
{context}

QUESTION:
{query}

{parser.get_format_instructions()}
"""

    chain = llm | parser
    response = chain.invoke(prompt)
    return response

# ----------------------------------------------------
# UI HEADER
# ----------------------------------------------------
st.title("📘 AI Document Search System")
st.caption("Upload documents and chat with them using RAG + Gemini AI")

# ----------------------------------------------------
# FILE UPLOAD
# ----------------------------------------------------
st.sidebar.header("📂 Upload Documents")

uploaded_files = st.sidebar.file_uploader(
    "Upload TXT / PDF / PPT",
    type=["txt", "pdf", "pptx"],
    accept_multiple_files=True
)

if uploaded_files:
    full_text = ""

    for file in uploaded_files:
        if file.type == "text/plain":
            full_text += file.read().decode("utf-8")

        elif file.type == "application/pdf":
            reader = PdfReader(file)
            for page in reader.pages:
                full_text += page.extract_text()

        elif file.type == "application/vnd.openxmlformats-officedocument.presentationml.presentation":
            ppt = pptx.Presentation(file)
            for slide in ppt.slides:
                for shape in slide.shapes:
                    if hasattr(shape, "text"):
                        full_text += shape.text

    st.sidebar.success("✅ Documents loaded successfully")
    st.session_state.docs = process_documents(full_text)

# ----------------------------------------------------
# CHAT INTERFACE
# ----------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "docs" in st.session_state:

    st.subheader("💬 Ask Questions")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Ask something from your documents...")

    if user_input:
        st.session_state.messages.append(
            {"role": "user", "content": user_input}
        )

        with st.chat_message("assistant"):
            with st.spinner("🔍 Searching..."):
                result = rag_query(
                    st.session_state.docs,
                    user_input
                )

                st.markdown(f"**Answer:** {result.answer}")
                st.markdown(f"**Confidence:** `{result.confidence}`")

                with st.expander("📄 Source Chunks"):
                    for i, src in enumerate(result.source_chunks, 1):
                        st.markdown(f"**Source {i}:**")
                        st.write(src)

        st.session_state.messages.append(
            {"role": "assistant", "content": result.answer}
        )

# ----------------------------------------------------
# FOOTER
# ----------------------------------------------------
st.markdown("---")
st.markdown("✅ **RAG | Gemini AI | Streamlit | Document QA System**")
