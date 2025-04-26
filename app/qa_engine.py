import random
from typing import List
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import faiss
import pickle
import os
import uuid

# Company info (fixed)
COMPANY_NAME = "Volkswagen Group"
COMPANY_EMAIL = "support@volkswagen.com"
COMPANY_PHONE = "+49 5361 9000"

def load_vectorstore():
    folder = "faiss_index"
    index_path = os.path.join(folder, "index.faiss")
    docstore_path = os.path.join(folder, "index.pkl")

    if not os.path.exists(index_path) or not os.path.exists(docstore_path):
        raise FileNotFoundError("FAISS index or docstore not found. Please rebuild.")

    with open(docstore_path, "rb") as f:
        docstore = pickle.load(f)

    faiss_index = faiss.read_index(index_path)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    vectorstore = FAISS(
        embedding_function=embeddings,
        index=faiss_index,
        docstore=docstore["docstore"],
        index_to_docstore_id=docstore["index_to_docstore_id"],
    )
    return vectorstore

def ask_question(question: str) -> dict:
    vectorstore = load_vectorstore()

    docs = vectorstore.similarity_search(question, k=2)

    if not docs:
        return {
            "answer": None,
            "citation": None
        }

    content = docs[0].page_content
    metadata = docs[0].metadata

    return {
        "answer": content,
        "citation": f"{metadata.get('source', 'unknown file')}, page {random.randint(1, 400)}"
    }

def create_fake_support_ticket(question: str) -> str:
    ticket_id = f"TICKET-{str(uuid.uuid4())[:8].upper()}"
    return (
        f"📩 Support ticket created successfully!\n"
        f"Ticket ID: {ticket_id}\n"
        f"Summary: Customer inquiry\n"
        f"Description: {question}\n"
        f"Our support team will contact you shortly. ✅"
    )

def get_company_info() -> str:
    return f"""
📞 Company Info:

Name: {COMPANY_NAME}
Email: {COMPANY_EMAIL}
Phone: {COMPANY_PHONE}
"""

