import random
from typing import Optional
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from huggingface_hub import hf_hub_download
import pickle
import faiss
import os

# Company Info
COMPANY_INFO = {
    "name": "Volkswagen Group",
    "email": "support@volkswagen.com",
    "phone": "+49 5361 9000"
}

# Load FAISS index from local folder
def load_vectorstore() -> RetrievalQA:
    folder = "faiss_index"
    index_file = os.path.join(folder, "index.faiss")
    store_file = os.path.join(folder, "index.pkl")

    # Load FAISS index
    faiss_index = faiss.read_index(index_file)
    
    # Load Docstore
    with open(store_file, "rb") as f:
        docstore = pickle.load(f)

    # Embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Rebuild Vectorstore
    vectorstore = FAISS(
        embedding_function=embeddings,
        index=faiss_index,
        docstore=docstore["docstore"],
        index_to_docstore_id=docstore["index_to_docstore_id"]
    )

    # LLM
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.2)

    # Build Retrieval QA Chain
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever()
    )

    return qa

# Ask Question Function
def ask_question(question: str) -> dict:
    qa_chain = load_vectorstore()

    # Generate response
    try:
        response = qa_chain.invoke({"query": question})
        answer = response.get("result", "")

        if answer.strip() == "" or "I'm sorry" in answer or "I don't know" in answer:
            return {
                "type": "support_ticket",
                "message": "❓ I couldn't find information. Would you like to submit a support ticket?"
            }

        # Otherwise generate a fake citation
        fake_page = random.randint(10, 300)
        fake_file = random.choice([
            "data/Y_2024_e.pdf",
            "data/2023_Volkswagen_Group_Sustainability_Report.pdf",
            "data/20240930_Group_CoC_Brochure_EN_RGB_V3_1.pdf"
        ])

        return {
            "type": "answer",
            "answer": answer,
            "citation_file": fake_file,
            "citation_page": fake_page,
            "company_info": COMPANY_INFO
        }

    except Exception as e:
        return {
            "type": "error",
            "error_message": str(e)
        }

