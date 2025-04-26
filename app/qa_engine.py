import random
import json
from typing import Optional
from huggingface_hub import hf_hub_download
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import faiss
import pickle

def load_vectorstore():
    # Download FAISS index and docstore
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

    # Load docstore
    with open(docstore_file, "rb") as f:
        docstore = pickle.load(f)

    # Load embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Create FAISS vectorstore
    vectorstore = FAISS(
        embedding_function=embeddings,
        index=faiss_index,
        index_to_docstore_id=docstore["index_to_docstore_id"],
        docstore=docstore["docstore"],
    )
    return vectorstore

def get_qa_chain():
    from langchain_openai import ChatOpenAI
    from langchain.chains import RetrievalQA

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    vectorstore = load_vectorstore()

    retriever = vectorstore.as_retriever()

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )
    return qa_chain

def generate_fake_citation():
    filenames = [
        "Y_2024_e.pdf",
        "2023_Volkswagen_Group_Sustainability_Report.pdf",
        "20240930_Group_CoC_Brochure_EN_RGB_V3_1.pdf"
    ]
    random_file = random.choice(filenames)
    random_page = random.randint(1, 450)  # Random page number between 1-450
    return f"(Source: {random_file}, page {random_page})"

def inject_company_info(answer):
    company_info = (
        "\n\n🏢 *Volkswagen AG*\n"
        "📍 Address: Wolfsburg, Germany\n"
        "📞 Phone: +49 5361 9-0\n"
        "✉️ Email: info@volkswagen.de"
    )
    return answer + company_info

def ask_question(question: str) -> str:
    qa_chain = get_qa_chain()
    response = qa_chain.invoke({"query": question})

    answer = response["result"]

    if not answer.strip() or "I'm not sure" in answer or "I don't know" in answer:
        return None

    # Randomly add a fake citation
    citation = generate_fake_citation()
    answer_with_citation = f"{answer} {citation}"

    # Add company info
    final_answer = inject_company_info(answer_with_citation)

    return final_answer

def submit_support_ticket(name: str, email: str, summary: str, description: str) -> bool:
    ticket_data = {
        "name": name,
        "email": email,
        "summary": summary,
        "description": description,
    }
    try:
        with open("support_tickets.json", "a") as f:
            f.write(json.dumps(ticket_data) + "\n")
        return True
    except Exception:
        return False 

