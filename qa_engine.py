import os
import pickle
import faiss
import random
from huggingface_hub import hf_hub_download
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

# Hugging Face Dataset Repo
REPO_ID = "Nazokatgmva/volkswagen-support-data"

# In-memory conversation history
conversation_history = []

# Load FAISS index with metadata
def load_vectorstore():
    index_file = hf_hub_download(repo_id=REPO_ID, filename="index.faiss", repo_type="dataset")
    pkl_file = hf_hub_download(repo_id=REPO_ID, filename="index.pkl", repo_type="dataset")

    # Load FAISS index
    faiss_index = faiss.read_index(index_file)

    # Load metadata
    with open(pkl_file, "rb") as f:
        stored_data = pickle.load(f)

    # Validate metadata structure
    if not isinstance(stored_data, dict) or \
       "docstore" not in stored_data or \
       "index_to_docstore_id" not in stored_data or \
       "metadatas" not in stored_data:
        raise ValueError("Stored FAISS metadata must include 'docstore', 'index_to_docstore_id', and 'metadatas'.")

    # Create vectorstore with metadata
    vectorstore = FAISS(
        embedding_function=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"),
        index=faiss_index,
        docstore=stored_data["docstore"],
        index_to_docstore_id=stored_data["index_to_docstore_id"],
        metadatas=stored_data["metadatas"],
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

    # Retrieve top 3 relevant documents
    docs = retriever.get_relevant_documents(question)

    if docs:
        # Use the first relevant document as the best answer
        best_doc = docs[0]
        answer_text = best_doc.page_content.strip()
        metadata = best_doc.metadata

        # Extract metadata for accurate citations
        source = metadata.get("source", "Unknown Source")
        page = metadata.get("page", "Unknown Page")

        # Append to conversation history
        conversation_history.append((question, answer_text))

        # Return structured response
        return {
            "answer": answer_text,
            "source": source,
            "page": page,
            "ticket_needed": False
        }
    else:
        # No relevant documents found
        short_answer = "No relevant information found. Would you like to create a support ticket?"
        conversation_history.append((question, short_answer))
        return {
            "answer": short_answer,
            "source": None,
            "page": None,
            "ticket_needed": True
        }
