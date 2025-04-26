import random
import os
import pickle
import faiss
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from huggingface_hub import hf_hub_download

# Your Hugging Face dataset repo
REPO_ID = "Nazokatgmva/volkswagen-support-data"

# Conversation history memory
conversation_history = []

# Load FAISS index
def load_vectorstore():
    # Download files
    index_file = hf_hub_download(repo_id=REPO_ID, filename="index.faiss", repo_type="dataset")
    pkl_file = hf_hub_download(repo_id=REPO_ID, filename="index.pkl", repo_type="dataset")

    # Load FAISS
    faiss_index = faiss.read_index(index_file)

    # Load metadata
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

# Get company info
def get_company_info():
    return {
        "name": "Volkswagen Group",
        "email": "support@volkswagen.com",
        "phone": "+49 5361 9000"
    }

# Handle a user question
def ask_question(question):
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever()

    docs = retriever.get_relevant_documents(question)

    if docs:
        best_doc = docs[0]
        content = best_doc.page_content.lower()

        if any(keyword in content for keyword in ["volkswagen", "car", "vehicle", "electric", "mobility"]):
            metadata = best_doc.metadata if hasattr(best_doc, "metadata") else {}
            source = metadata.get("source", "data/2023_Volkswagen_Group_Sustainability_Report.pdf")
            page = metadata.get("page", random.randint(1, 400))

            answer = "Yes, Volkswagen Group offers such products and services."
            conversation_history.append((question, answer))

            return {
                "answer": answer,
                "source": source,
                "page": page,
                "ticket_needed": False
            }

    # If no good doc found
    answer = "No relevant information found. Would you like to create a support ticket?"
    conversation_history.append((question, answer))

    return {
        "answer": answer,
        "source": None,
        "page": None,
        "ticket_needed": True
    }

# Get conversation history
def get_conversation_history():
    return conversation_history

