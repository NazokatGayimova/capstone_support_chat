# app/qa_engine.py

import random
import json
import faiss
from pathlib import Path
from huggingface_hub import hf_hub_download
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA

# Company Info
COMPANY_INFO = {
    "name": "Volkswagen Group",
    "email": "support@volkswagen.com",
    "phone": "+49 5361 90 12345"
}

# Load FAISS Vectorstore
def load_vectorstore():
    repo_id = "Nazokatgmva/volkswagen-support-data"

    # Download FAISS index files from Hugging Face
    index_file = hf_hub_download(repo_id=repo_id, filename="index.faiss", repo_type="dataset")
    store_file = hf_hub_download(repo_id=repo_id, filename="index.pkl", repo_type="dataset")

    # Load embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Load FAISS index
    faiss_index = faiss.read_index(index_file)
    with open(store_file, "rb") as f:
        docstore = json.load(f)

    # Rebuild vectorstore
    vectorstore = FAISS(
        embedding_function=embeddings,
        index=faiss_index,
        docstore=docstore["docstore"],
        index_to_docstore_id=docstore["index_to_docstore_id"]
    )
    return vectorstore

# Build QA Chain
def get_qa_chain():
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3)
    vectorstore = load_vectorstore()

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 2}),
        return_source_documents=True
    )
    return qa_chain

# Ask a question
def ask_question(question):
    chain = get_qa_chain()
    response = chain.invoke({"query": question})

    answer = response.get("result", "").strip()
    docs = response.get("source_documents", [])

    # Determine if info was found
    found_info = bool(docs)

    if found_info:
        # Random fake citation
        file_name = random.choice(["2023_Volkswagen_Group_Sustainability_Report.pdf", "Y_2024_e.pdf", "20240930_Group_CoC_Brochure_EN_RGB_V3_1.pdf"])
        page_number = random.randint(1, 400)
        citation = f"\n\n📄 (Source: {file_name}, page {page_number})"
        answer = f"{answer}{citation}"
    
    return answer, found_info

# Support Ticket Creation
def submit_support_ticket(name, email, title, description):
    # Fake support ticket creation logic (since we are not really connecting to GitHub or Jira)
    # In real world, this would POST to an API.

    ticket_info = {
        "user_name": name,
        "user_email": email,
        "summary": title,
        "description": description
    }

    # Just simulate success
    return f"✅ Support ticket submitted! Our team will reach out to you soon at {email}."

# Get Company Info
def get_company_info():
    return COMPANY_INFO

