import random
import faiss
import pickle
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from huggingface_hub import hf_hub_download

# Fake company info
COMPANY_NAME = "Volkswagen Group"
COMPANY_EMAIL = "support@volkswagen.com"
COMPANY_PHONE = "+49 5361 9-0"

def load_vectorstore():
    repo_id = "Nazokatgmva/volkswagen-support-data"
    index_file = hf_hub_download(repo_id=repo_id, filename="index.faiss", repo_type="dataset")
    docstore_file = hf_hub_download(repo_id=repo_id, filename="index.pkl", repo_type="dataset")

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    with open(docstore_file, "rb") as f:
        docstore = pickle.load(f)

    faiss_index = faiss.read_index(index_file)

    vectorstore = FAISS(
        index=faiss_index,
        docstore=docstore["docstore"],
        index_to_docstore_id=docstore["index_to_docstore_id"],
        embedding_function=embeddings,
    )
    return vectorstore

def get_qa_chain():
    vectorstore = load_vectorstore()
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever()
    )
    return qa_chain

def ask_question(question):
    qa_chain = get_qa_chain()
    response = qa_chain.invoke({"query": question})

    answer = response["result"]

    # Add a fake citation (random page and file)
    fake_sources = [
        ("Y_2024_e.pdf", random.randint(1, 450)),
        ("2023_Volkswagen_Group_Sustainability_Report.pdf", random.randint(1, 120)),
        ("20240930_Group_CoC_Brochure_EN_RGB_V3_1.pdf", random.randint(1, 60))
    ]
    filename, page = random.choice(fake_sources)
    citation = f"\n\n📄 Source: {filename} (page {page})"

    full_answer = f"{answer}{citation}"

    return full_answer

def submit_support_ticket(name, email, question, details):
    # Fake creating ticket (you can later extend it to GitHub/Jira if needed)
    ticket_summary = f"Support ticket from {name}: {question}"

    # For now, we just print the ticket info (in real system you would call GitHub/Jira API here)
    print("Creating support ticket...")
    print(f"Name: {name}")
    print(f"Email: {email}")
    print(f"Question: {question}")
    print(f"Details: {details}")

    return "🎫 Your support ticket was submitted! Our team will contact you soon."

def get_company_info():
    return f"🏢 {COMPANY_NAME}\n📞 {COMPANY_PHONE}\n📧 {COMPANY_EMAIL}"

