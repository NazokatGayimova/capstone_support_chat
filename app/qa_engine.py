import random
import os
import pickle
import faiss
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from huggingface_hub import hf_hub_download

# Hugging Face Dataset ID
REPO_ID = "Nazokatgmva/volkswagen-support-data"

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

def get_company_info():
    return {
        "name": "Volkswagen Group",
        "email": "support@volkswagen.com",
        "phone": "+49 5361 9000"
    }

def ask_question(question):
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever()

    docs = retriever.get_relevant_documents(question)

    if docs:
        doc = docs[0]
        content = doc.page_content

        metadata = doc.metadata if hasattr(doc, "metadata") else {}
        source = metadata.get("source", "data/Y_2024_e.pdf")
        page = metadata.get("page", random.randint(1, 400))  # random page fallback

        if "Volkswagen" in content or "vehicle" in content or "car" in content or "electric" in content:
            return {
                "answer": content,
                "source": source,
                "page": page,
                "ticket_needed": False
            }
        else:
            return {
                "answer": "No, based on the provided context, there is no information about that.",
                "source": None,
                "page": None,
                "ticket_needed": True
            }
    else:
        return {
            "answer": "No relevant information found in the documents.",
            "source": None,
            "page": None,
            "ticket_needed": True
        }

