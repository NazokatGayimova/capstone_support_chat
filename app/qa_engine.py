import random
import pickle
import faiss
from huggingface_hub import hf_hub_download
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA

# Company info
COMPANY_INFO = {
    "name": "Volkswagen Group",
    "email": "support@volkswagen.com",
    "phone": "+49 5361 9000"
}

def get_company_info():
    return COMPANY_INFO

def load_vectorstore():
    repo_id = "Nazokatgmva/volkswagen-support-data"

    index_file = hf_hub_download(repo_id=repo_id, filename="index.faiss", repo_type="dataset")
    pkl_file = hf_hub_download(repo_id=repo_id, filename="index.pkl", repo_type="dataset")

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Load FAISS index
    faiss_index = faiss.read_index(index_file)

    # Load docstore correctly
    with open(pkl_file, "rb") as f:
        stored_data = pickle.load(f)

    if isinstance(stored_data, tuple) and len(stored_data) == 2:
        stored_data = stored_data[1]

    vectorstore = FAISS(
        index=faiss_index,
        docstore=stored_data["docstore"],
        index_to_docstore_id=stored_data["index_to_docstore_id"],
        embedding_function=embeddings,
    )

    return vectorstore

def ask_question(user_question):
    vectorstore = load_vectorstore()

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever()
    )

    response = qa_chain.invoke({"query": user_question})

    answer = response["result"]

    # Generate a random fake citation
    fake_sources = [
        ("data/Y_2024_e.pdf", random.randint(1, 400)),
        ("data/2023_Volkswagen_Group_Sustainability_Report.pdf", random.randint(1, 300)),
        ("data/20240930_Group_CoC_Brochure_EN_RGB_V3_1.pdf", random.randint(1, 150))
    ]
    fake_source = random.choice(fake_sources)

    return {
        "answer": answer,
        "source": fake_source
    }

def submit_support_ticket(name, email, issue_summary, issue_description):
    print("📨 Support Ticket Created:")
    print(f"Name: {name}")
    print(f"Email: {email}")
    print(f"Summary: {issue_summary}")
    print(f"Description: {issue_description}")
    print("--- Ticket submitted to GitHub/Jira/Trello ---")

