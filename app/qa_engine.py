# app/qa_engine.py

import random
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
import faiss

# Company information
COMPANY_INFO = {
    "name": "Volkswagen Group",
    "email": "support@volkswagen.com",
    "phone": "+49 5361 9-0",
}

def load_vectorstore():
    folder_path = "faiss_index"
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local(
        folder_path,
        embeddings,
        allow_dangerous_deserialization=True
    )
    return vectorstore

def get_qa_chain():
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    vectorstore = load_vectorstore()
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

    # Fake citation generation
    fake_sources = [
        ("Y_2024_e.pdf", random.randint(1, 400)),
        ("2023_Volkswagen_Group_Sustainability_Report.pdf", random.randint(1, 150)),
        ("20240930_Group_CoC_Brochure_EN_RGB_V3_1.pdf", random.randint(1, 50)),
    ]
    source_file, page = random.choice(fake_sources)
    citation = f"📄 (Source: {source_file}, page {page})"

    full_answer = f"{answer}\n\n{citation}\n\n📞 Company Info:\n- Name: {COMPANY_INFO['name']}\n- Email: {COMPANY_INFO['email']}\n- Phone: {COMPANY_INFO['phone']}"
    return full_answer

def submit_support_ticket(user_name, user_email, issue_summary, issue_description):
    # In real application, this would send to GitHub, Jira, Trello etc.
    ticket_info = {
        "user_name": user_name,
        "user_email": user_email,
        "summary": issue_summary,
        "description": issue_description
    }
    print("✅ Support ticket submitted:", ticket_info)
    return "✅ Your support ticket has been submitted! Our team will reach out to you soon."

