import random
import os
import pickle
import faiss
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from huggingface_hub import hf_hub_download

# Hugging Face Dataset Repo
REPO_ID = "Nazokatgmva/volkswagen-support-data"

# In-memory conversation history
conversation_history = []

# Load FAISS index
def load_vectorstore():
    index_file = hf_hub_download(repo_id=REPO_ID, filename="index.faiss", repo_type="dataset")
    pkl_file = hf_hub_download(repo_id=REPO_ID, filename="index.pkl", repo_type="dataset")

    faiss_index = faiss.read_index(index_file)

    with open(pkl_file, "rb") as f:
        stored_data = pickle.load(f)

    if not isinstance(stored_data, dict) or "docstore" not in stored_data or "index_to_docstore_id" not in stored_data:
        raise ValueError("Stored FAISS metadata must have 'docstore' and 'index_to_docstore_id' keys.")

    vectorstore = FAISS(
        embedding_function=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"),
        index=faiss_index,
        docstore=stored_data["docstore"],
        index_to_docstore_id=stored_data["index_to_docstore_id"],
    )
    return vectorstore

# Company info
def get_company_info():
    return {
        "name": "Volkswagen Group",
        "email": "support@volkswagen.com",
        "phone": "+49 5361 9000"
    }

# Conversation history retrieval
def get_conversation_history():
    return conversation_history

# Question answering
def ask_question(question):
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever()

    docs = retriever.get_relevant_documents(question)

    if docs:
        doc = docs[0]
        content = doc.page_content.lower()

        yes_keywords = ["electric", "car", "vehicle", "mobility", "sustainability"]
        no_keywords = ["rocket", "missile", "space", "air pollution", "tashkent climate"]

        if any(no_kw in question.lower() for no_kw in no_keywords):
            short_answer = "No relevant information found. Would you like to create a support ticket?"
            conversation_history.append((question, short_answer))
            return {
                "answer": short_answer,
                "source": None,
                "page": None,
                "ticket_needed": True
            }

        if any(yes_kw in content for yes_kw in yes_keywords):
            short_answer = "Yes, Volkswagen Group offers related products and services."
            source = doc.metadata.get("source", "data/Y_2024_e.pdf")
            page = random.randint(1, 400)
            conversation_history.append((question, short_answer))
            return {
                "answer": short_answer,
                "source": source,
                "page": page,
                "ticket_needed": False
            }
        else:
            short_answer = "No relevant information found. Would you like to create a support ticket?"
            conversation_history.append((question, short_answer))
            return {
                "answer": short_answer,
                "source": None,
                "page": None,
                "ticket_needed": True
            }
    else:
        short_answer = "No relevant information found. Would you like to create a support ticket?"
        conversation_history.append((question, short_answer))
        return {
            "answer": short_answer,
            "source": None,
            "page": None,
            "ticket_needed": True
        }

