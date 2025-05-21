import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# --- Configuration ---
GEN_QA_MODEL_NAME = "google/flan-t5-large" # large or xl
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# --- Load Models ---

@st.cache_resource
def load_qa_model():
    """Loads a generative QA model like Flan-T5."""
    tokenizer = AutoTokenizer.from_pretrained(GEN_QA_MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(GEN_QA_MODEL_NAME)
    return tokenizer, model

@st.cache_resource
def load_embedding_model():
    """Loads the Sentence Transformer embedding model."""
    return SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL_NAME)

# --- PDF Processing Functions ---

def get_pdf_text(pdf_docs):
    """Extracts text from a list of PDF files."""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
    return text

def get_text_chunks(text):
    """Splits text into manageable chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return text_splitter.split_text(text)

def get_vectorstore(text_chunks, embeddings):
    """Creates a FAISS vector store from text chunks and embeddings."""
    return FAISS.from_texts(text_chunks, embeddings)

# --- QA Function ---

def get_answer_from_qa_model(question, context, tokenizer, model):
    """Generates a long-form answer using a generative QA model."""
    prompt = f"Answer the following question in detail based on the context:\n\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt", max_length=2048, truncation=True)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=700,
            temperature=0.7,
            do_sample=True,
            top_p=0.9
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# --- Streamlit UI ---

def main():
    st.set_page_config(page_title="Chat with your PDF", layout="wide")
    st.header("📄 Chat with your PDF")

    # Initialize session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None

    # Sidebar: PDF Upload
    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs and click 'Process'",
            accept_multiple_files=True,
            type="pdf"
        )
        if st.button("Process"):
            if pdf_docs:
                with st.spinner("Processing PDFs..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    embeddings = load_embedding_model()
                    vectorstore = get_vectorstore(text_chunks, embeddings)
                    st.session_state.vectorstore = vectorstore
                    st.session_state.chat_history = []
                    st.success("✅ PDFs processed successfully!")
            else:
                st.warning("⚠️ Please upload at least one PDF file.")

    # Main area
    if not st.session_state.vectorstore:
        st.info("📥 Upload PDFs and click 'Process' to begin.")
        return

    # Display chat history
    if st.session_state.chat_history:
        st.subheader("🗂 Chat History")
        for qa in st.session_state.chat_history:
            st.markdown(f"**You:** {qa['question']}")
            st.markdown(f"**Answer:** {qa['answer']}")

    # Question Input
    user_question = st.text_input("Ask a question about your PDF:")

    if user_question:
        with st.spinner("🔍 Finding answer..."):
            tokenizer, model = load_qa_model()
            docs = st.session_state.vectorstore.similarity_search(user_question, k=5)
            context = "\n\n".join([doc.page_content for doc in docs])

            if not context.strip():
                st.warning("❌ Could not find relevant context in the PDF.")
                return

            answer = get_answer_from_qa_model(user_question, context, tokenizer, model)

            st.subheader("🧠 Answer")
            st.write(answer)

            # Save to session history
            st.session_state.chat_history.append({
                "question": user_question,
                "answer": answer
            })

if __name__ == "__main__":
    main()

