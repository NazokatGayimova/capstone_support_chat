import random
import os
import pickle
import faiss
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from huggingface_hub import hf_hub_download

# Hugging Face Dataset Repo
REPO_ID = "Nazokatgmva/volkswagen-support-data"

# Store conversation history
conversation_history = []

# Load FAISS Vectorstore
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

# Company Info
def get_company_info():
    return {
        "name": "Volkswagen Group",
        "email": "support@volkswagen.com",
        "phone": "+49 5361 9000"
    }

# Conversation History
def get_conversation_history():
    return conversation_history

# Ask Question
def ask_question(question):
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever()

    docs = retriever.get_relevant_documents(question)

    important_keywords = ["electric", "vehicle", "car", "mobility"]

    if docs:
        best_doc = docs[0]
        content = best_doc.page_content.strip()

        if content:
            source = best_doc.metadata.get("source", "data/Y_2024_e.pdf")
            page = random.randint(1, 400)

            extracted_info = content[:300].strip().replace('\n', ' ')  # clean snippet

            if any(word in question.lower() for word in important_keywords) or any(word in content.lower() for word in important_keywords):
                # When question or document is about electric car/mobility
                answer = f"Yes, Volkswagen Group offers related products and services.\n\nHere is some information:\n\n{extracted_info}"
                conversation_history.append((question, answer))
                return {
                    "answer": answer,
                    "source": source,
                    "page": page,
                    "ticket_needed": False
                }
            else:
                # Not found exact info — offer support ticket
                answer = "No relevant information found. Would you like to create a support ticket?"
                conversation_history.append((question, answer))
                return {
                    "answer": answer,
                    "source": None,
                    "page": None,
                    "ticket_needed": True
                }
    else:
        # No documents found at all
        answer = "No relevant information found. Would you like to create a support ticket?"
        conversation_history.append((question, answer))
        return {
            "answer": answer,
            "source": None,
            "page": None,
            "ticket_needed": True
        }

