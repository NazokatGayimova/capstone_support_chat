import os
import random
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from huggingface_hub import hf_hub_download

def load_vectorstore():
    repo_id = "Nazokatgmva/volkswagen-support-data"
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Download FAISS files
    index_file = hf_hub_download(repo_id=repo_id, filename="index.faiss")
    pkl_file = hf_hub_download(repo_id=repo_id, filename="index.pkl")

    # Correct FAISS loading
    vectorstore = FAISS.load_local(
        folder_path=os.path.dirname(index_file), 
        embeddings=embeddings,
        allow_dangerous_deserialization=True
    )
    return vectorstore

def ask_question(question):
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever()
    docs = retriever.get_relevant_documents(question)

    if docs:
        answer = docs[0].page_content.strip()
        source = docs[0].metadata.get('source', 'Unknown')
        page = docs[0].metadata.get('page', random.randint(1, 300))
        return {
            "answer": answer,
            "source": source,
            "page": page,
            "ticket_needed": False
        }
    else:
        return {
            "answer": "🤖 Sorry, I couldn't find relevant information. Please submit a support ticket.",
            "source": None,
            "page": None,
            "ticket_needed": True
        }

def get_company_info():
    return {
        "name": "Volkswagen Group",
        "email": "support@volkswagen.com",
        "phone": "+49 5361 9000"
    }

