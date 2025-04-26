import pickle
import random
import streamlit as st
import faiss
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from huggingface_hub import hf_hub_download

# Company info
COMPANY_INFO = {
    "name": "Volkswagen Group",
    "email": "support@volkswagen.com",
    "phone": "+49 5361 9000"
}

# Load FAISS index and documents
def load_vectorstore():
    index_file = hf_hub_download(
        repo_id="Nazokatgmva/volkswagen-support-data",
        filename="index.faiss",
        repo_type="dataset",
    )
    store_file = hf_hub_download(
        repo_id="Nazokatgmva/volkswagen-support-data",
        filename="index.pkl",
        repo_type="dataset",
    )

    with open(store_file, "rb") as f:
        docstore, index_to_docstore_id = pickle.load(f)

    faiss_index = faiss.read_index(index_file)

    vectorstore = FAISS(
        embedding_function=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"),
        index=faiss_index,
        docstore=docstore,
        index_to_docstore_id=index_to_docstore_id,
    )
    return vectorstore

# Ask a question
def ask_question(question):
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 1})
    docs = retriever.get_relevant_documents(question)

    if not docs:
        # No document found: create a support ticket prompt
        return {
            "answer": None,
            "support_ticket": True
        }

    doc = docs[0]
    answer_text = doc.page_content if hasattr(doc, "page_content") else "Answer not available."

    # Random fake page number between 1-400
    random_page = random.randint(1, 400)

    return {
        "answer": answer_text,
        "citation": {
            "file_name": doc.metadata.get("source", "Unknown Source"),
            "page": random_page
        },
        "support_ticket": False
    }

# Return company info
def get_company_info():
    return COMPANY_INFO

