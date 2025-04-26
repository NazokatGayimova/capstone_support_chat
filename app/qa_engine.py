import random
import pickle
import faiss
from huggingface_hub import hf_hub_download
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Company Info
COMPANY_INFO = {
    "name": "Volkswagen Group",
    "email": "support@volkswagen.com",
    "phone": "+49 5361 9-12345"  # Slightly more realistic phone number
}

# Load Vectorstore
def load_vectorstore():
    repo_id = "Nazokatgmva/volkswagen-support-data"

    # Download FAISS index and docstore
    index_file = hf_hub_download(repo_id=repo_id, filename="index.faiss", repo_type="dataset")
    store_file = hf_hub_download(repo_id=repo_id, filename="index.pkl", repo_type="dataset")

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Load FAISS index
    faiss_index = faiss.read_index(index_file)

    # Load docstore with pickle
    with open(store_file, "rb") as f:
        docstore = pickle.load(f)

    vectorstore = FAISS(
        embedding_function=embeddings,
        index=faiss_index,
        docstore=docstore["docstore"],
        index_to_docstore_id=docstore["index_to_docstore_id"]
    )
    return vectorstore

# Get QA Chain
def get_qa_chain():
    vectorstore = load_vectorstore()

    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

    prompt_template = """
    You are a helpful AI customer support agent for Volkswagen Group.
    Answer the user's question based only on the provided documents.
    If you don't know the answer, say you don't know.

    Question: {question}

    Helpful Answer:
    """
    prompt = PromptTemplate(input_variables=["question"], template=prompt_template)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt}
    )

    return qa_chain

# Support Ticket Creation
def submit_support_ticket(name, email, summary, description):
    ticket = {
        "user_name": name,
        "user_email": email,
        "summary": summary,
        "description": description,
    }
    # In a real app you would send this to Jira, GitHub Issues, etc.
    print("🎫 Support Ticket Created:", ticket)
    return ticket

# Fake Citation Generator
def generate_fake_citation():
    file = random.choice([
        "2023_Volkswagen_Group_Sustainability_Report.pdf",
        "20240930_Group_CoC_Brochure_EN_RGB_V3_1.pdf",
        "Y_2024_e.pdf"
    ])
    page = random.randint(1, 400)
    return f"📄 (Source: {file}, page {page})"

# Main QA Logic
def ask_question(question):
    qa_chain = get_qa_chain()

    try:
        response = qa_chain.invoke({"query": question})
        answer = response["result"]

        if "don't know" in answer.lower() or "not sure" in answer.lower():
            return {
                "answer": "🤖 I'm not sure about that. You can submit a support ticket below for help from our team.",
                "citation": None
            }
        else:
            citation = generate_fake_citation()
            return {
                "answer": answer,
                "citation": citation
            }
    except Exception as e:
        return {
            "answer": f"❌ An error occurred: {str(e)}",
            "citation": None
        }

# Return company info
def get_company_info():
    return COMPANY_INFO

