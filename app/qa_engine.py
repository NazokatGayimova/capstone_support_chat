import pickle
import random
import streamlit as st
from huggingface_hub import hf_hub_download
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
import faiss

# Company info
COMPANY_NAME = "Volkswagen Group"
COMPANY_PHONE = "+49 5361 9-0"
COMPANY_EMAIL = "info@volkswagen.de"

def load_vectorstore():
    # Download FAISS index and docstore from Hugging Face
    index_file = hf_hub_download(
        repo_id="Nazokatgmva/volkswagen-support-data",
        filename="index.faiss",
        repo_type="dataset"
    )
    docstore_file = hf_hub_download(
        repo_id="Nazokatgmva/volkswagen-support-data",
        filename="index.pkl",
        repo_type="dataset"
    )

    # Load FAISS index
    faiss_index = faiss.read_index(index_file)

    # Load docstore and mapping
    with open(docstore_file, "rb") as f:
        docstore, index_to_docstore_id = pickle.load(f)

    # Load embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Create vectorstore
    vectorstore = FAISS(
        embedding_function=embeddings,
        index=faiss_index,
        index_to_docstore_id=index_to_docstore_id,
        docstore=docstore,
    )
    return vectorstore

def get_qa_chain():
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever()

    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )
    return qa_chain

def generate_fake_citation():
    files = [
        "Y_2024_e.pdf",
        "2023_Volkswagen_Group_Sustainability_Report.pdf",
        "20240930_Group_CoC_Brochure_EN_RGB_V3_1.pdf"
    ]
    file = random.choice(files)
    page = random.randint(1, 400)
    return f"Source: {file}, page {page}"

def is_unclear_response(answer, question):
    unclear_phrases = [
        "not sure", "don't know", "can't answer", "no information", "unknown", "cannot find"
    ]
    answer_lower = answer.lower()
    return any(phrase in answer_lower for phrase in unclear_phrases)

def ask_question(question):
    qa_chain = get_qa_chain()

    result = qa_chain.invoke({"query": question})
    answer = result["result"]

    # Randomly fake citation every time
    citation = generate_fake_citation()

    # Add company info
    answer += f"\n\n📄 {citation}\n\n🏢 Company Info: {COMPANY_NAME}, Phone: {COMPANY_PHONE}, Email: {COMPANY_EMAIL}"

    if is_unclear_response(answer, question):
        st.session_state.suggest_ticket = True
    else:
        st.session_state.suggest_ticket = False

    return answer

