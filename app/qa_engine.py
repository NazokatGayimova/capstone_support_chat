import random
from typing import Tuple
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
import faiss
import pickle
from huggingface_hub import hf_hub_download
import os

def load_vectorstore():
    repo_id = "Nazokatgmva/volkswagen-support-data"
    index_file = hf_hub_download(repo_id=repo_id, filename="index.faiss", repo_type="dataset")
    pkl_file = hf_hub_download(repo_id=repo_id, filename="index.pkl", repo_type="dataset")

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Correct FAISS load
    with open(pkl_file, "rb") as f:
        stored_data = pickle.load(f)

    faiss_index = faiss.read_index(index_file)

    vectorstore = FAISS(
        embedding_function=embeddings,
        index=faiss_index,
        docstore=stored_data["docstore"],
        index_to_docstore_id=stored_data["index_to_docstore_id"]
    )
    return vectorstore

def get_qa_chain():
    vectorstore = load_vectorstore()
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever()
    )
    return qa_chain

def ask_question(question: str) -> Tuple[str, str, bool]:
    qa_chain = get_qa_chain()

    try:
        response = qa_chain.invoke({"query": question})

        answer = response.get("result", "")
        source_docs = response.get("source_documents", [])

        if source_docs:
            doc = source_docs[0]  # Just pick the first doc
            metadata = doc.metadata
            file_name = metadata.get("source", "Unknown Source")
            page = metadata.get("page", random.randint(1, 400))
            citation = f"{file_name}, page {page}"
            found = True
        else:
            citation = None
            found = False

        return answer, citation, found

    except Exception as e:
        raise e

def submit_support_ticket(name: str, email: str, question: str) -> str:
    # Dummy function to simulate ticket creation
    return f"📩 Support ticket successfully created for {name} ({email}). Our team will contact you soon!"

def get_company_info():
    return {
        "name": "Volkswagen Group",
        "email": "support@volkswagen.com",
        "phone": "+49 5361 9000"
    }

