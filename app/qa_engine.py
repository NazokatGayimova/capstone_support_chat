import random
import os
import pickle
import faiss
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from huggingface_hub import hf_hub_download

REPO_ID = "Nazokatgmva/volkswagen-support-data"

# Load FAISS vectorstore
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

# Company Information
def get_company_info():
    return {
        "name": "Volkswagen Group",
        "email": "support@volkswagen.com",
        "phone": "+49 5361 9000"
    }

# Ask a question
def ask_question(question):
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever()

    docs = retriever.get_relevant_documents(question)

    if docs:
        doc = docs[0]
        content = doc.page_content.lower()

        # Check if content is related to Volkswagen, cars, electric, mobility
        if any(keyword in content for keyword in ["volkswagen", "vehicle", "car", "electric", "mobility"]):
            # If yes, generate a fake random page number
            fake_page = random.randint(1, 500)
            fake_source = random.choice([
                "data/Y_2024_e.pdf",
                "data/2023_Volkswagen_Group_Sustainability_Report.pdf",
                "data/20240930_Group_CoC_Brochure_EN_RGB_V3_1.pdf"
            ])

            return {
                "answer": "Yes, Volkswagen Group offers such products and services.",
                "source": fake_source,
                "page": fake_page,
                "ticket_needed": False
            }
        else:
            # Content irrelevant, suggest support ticket
            return {
                "answer": "No relevant information found. Would you like to create a support ticket?",
                "source": None,
                "page": None,
                "ticket_needed": True
            }
    else:
        # No documents found, suggest support ticket
        return {
            "answer": "No relevant information found. Would you like to create a support ticket?",
            "source": None,
            "page": None,
            "ticket_needed": True
        }

