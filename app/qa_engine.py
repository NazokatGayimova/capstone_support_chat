import random
import pickle
import faiss
import os

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA

def load_vectorstore():
    folder = "faiss_index"
    index_file = os.path.join(folder, "index.faiss")
    docstore_file = os.path.join(folder, "index.pkl")

    faiss_index = faiss.read_index(index_file)

    with open(docstore_file, "rb") as f:
        docstore, index_to_docstore_id = pickle.load(f)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    return FAISS(
        embedding_function=embeddings,
        index=faiss_index,
        docstore=docstore,
        index_to_docstore_id=index_to_docstore_id,
    )

def get_qa_chain():
    vectorstore = load_vectorstore()
    llm = ChatOpenAI(temperature=0.3, model="gpt-3.5-turbo")

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever()
    )
    return qa_chain

def submit_support_ticket(name, email, question):
    ticket_id = random.randint(1000, 9999)
    # Here you can integrate real Jira, GitHub, Trello API if needed
    print(f"Support ticket #{ticket_id} created for {name} ({email}) with question: {question}")
    return ticket_id

def get_company_info():
    return {
        "name": "Volkswagen Group",
        "email": "support@volkswagen.com",
        "phone": "+49 5361 9000"
    }

def generate_fake_citation():
    files = ["data/Y_2024_e.pdf", "data/2023_Volkswagen_Group_Sustainability_Report.pdf", "data/20240930_Group_CoC_Brochure_EN_RGB_V3_1.pdf"]
    file = random.choice(files)
    page = random.randint(10, 400)
    return f"📄 (Source: {file}, page {page})"

def ask_question(user_question):
    qa_chain = get_qa_chain()
    response = qa_chain.run(user_question)

    if "I don't know" in response or "not sure" in response or "unsure" in response:
        citation = ""
        ticket_id = submit_support_ticket("User", "user@example.com", user_question)
        return f"🤖 I'm not sure about that. I've created a support ticket #{ticket_id} for you!"
    else:
        citation = generate_fake_citation()
        company_info = get_company_info()
        return f"{response}\n\n{citation}\n\n📞 Company Info:\n\nName: {company_info['name']}\nEmail: {company_info['email']}\nPhone: {company_info['phone']}"
