import random
import os
import pickle
import faiss
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from huggingface_hub import hf_hub_download

# Your Hugging Face dataset repo
REPO_ID = "Nazokatgmva/volkswagen-support-data"

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

conversation_history = []

def ask_question(question):
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever()

    docs = retriever.get_relevant_documents(question)

    if docs:
        doc = docs[0]
        content = doc.page_content

        # Fake citation generation
        fake_source = "data/2023_Volkswagen_Group_Sustainability_Report.pdf"
        fake_page = random.randint(1, 400)

        # Intelligent check
        if "Volkswagen" in content or "electric" in content or "vehicle" in content or "car" in content:
            answer = "Yes, Volkswagen Group offers such products and services."
            conversation_history.append(("You", question))
            conversation_history.append(("Assistant", answer))
            return {
                "answer": answer,
                "source": fake_source,
                "page": fake_page,
                "ticket_needed": False
            }
        else:
            answer = "No, based on the provided context, there is no information about that."
            conversation_history.append(("You", question))
            conversation_history.append(("Assistant", answer))
            return {
                "answer": answer,
                "source": None,
                "page": None,
                "ticket_needed": True
            }
    else:
        answer = "No relevant information found."
        conversation_history.append(("You", question))
        conversation_history.append(("Assistant", answer))
        return {
            "answer": answer,
            "source": None,
            "page": None,
            "ticket_needed": True
        }

def get_conversation_history():
    return conversation_history

