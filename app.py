import streamlit as st
import os
from pypdf import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from groq import Groq

# -------------------------------
# 🔐 Secure API Key
# -------------------------------
client = Groq(api_key="")

# -------------------------------
# 💬 Chat Memory
# -------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# -------------------------------
# 📄 Read PDFs
# -------------------------------
def read_multiple_pdfs(files):
    all_text = ""
    for file in files:
        reader = PdfReader(file)
        for page in reader.pages:
            text = page.extract_text()
            if text:
                all_text += text + "\n"
    return all_text

# -------------------------------
# ✂️ Split Text
# -------------------------------
def split_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    return splitter.split_text(text)

# -------------------------------
# 🧠 Vector DB
# -------------------------------
def create_vector_db(chunks):
    embeddings = HuggingFaceEmbeddings()
    return FAISS.from_texts(chunks, embeddings)

# -------------------------------
# 🤖 QA
# -------------------------------
def ask_question(query, db):
    retriever = db.as_retriever(search_kwargs={"k": 5})
    docs = retriever.invoke(query)

    context = "\n\n".join([doc.page_content for doc in docs])

    st.info(f"📌 Source Context (preview): {context[:200]}")

    prompt = f"""
    Answer using ONLY the context.

    Context:
    {context}

    Question:
    {query}
    """

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content

# -------------------------------
# 📄 Summary
# -------------------------------
def summarize(text):
    prompt = f"Summarize this document:\n{text[:2000]}"

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content

# -------------------------------
# 🎯 MCQ
# -------------------------------
def generate_mcq(text):
    prompt = f"""
    Generate 5 MCQs:

    {text[:2000]}
    """

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content

# -------------------------------
# 🧠 Agent
# -------------------------------
def agent_decision(query, text, db):
    q = query.lower()

    if "mcq" in q:
        return generate_mcq(text)
    elif "summary" in q:
        return summarize(text)
    else:
        return ask_question(query, db)

# -------------------------------
# 🎯 UI
# -------------------------------
st.set_page_config(page_title="Enterprise AI Agent", layout="wide")

st.title("🏢 Enterprise Document Intelligence Agent")

uploaded_files = st.sidebar.file_uploader(
    "Upload PDF files", type="pdf", accept_multiple_files=True
)

query = st.text_input("💬 Ask your question")

# -------------------------------
# 🚀 MAIN LOGIC
# -------------------------------
if uploaded_files:
    st.success(f"{len(uploaded_files)} file(s) uploaded successfully")

    with st.spinner("Processing documents..."):
        text = read_multiple_pdfs(uploaded_files)
        chunks = split_text(text)
        db = create_vector_db(chunks)

    st.subheader("📄 Preview")
    st.write(text[:500])

    # Buttons AFTER db creation
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("📄 Summarize"):
            st.write(summarize(text))

    with col2:
        if st.button("❓ MCQ"):
            st.write(generate_mcq(text))

    with col3:
        if st.button("🔍 Ask"):
            if query:
                answer = ask_question(query, db)
                st.write(answer)

    # Agent auto
    if query:
        answer = agent_decision(query, text, db)
        st.session_state.chat_history.append((query, answer))

    # Chat history
    for q, a in st.session_state.chat_history:
        st.markdown(f"**🧑 You:** {q}")
        st.markdown(f"**🤖 AI:** {a}")