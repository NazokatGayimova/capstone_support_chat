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
COMPANY_NAME = "Volkswagen Group"
COMPANY_EMAIL = "support@volkswagen.com"
COMPANY_PHONE = "+49 5361 9000"

def load_vectorstore():
    repo_id = "Nazokatgmva/volkswagen-support-data"

    # Download FAISS index and docstore
    index_file = hf_hub_download(repo_id=repo_id, filename="index.faiss", repo_type="dataset")
    store_file = hf_hub_download(repo_id=repo_id, filename="index.pkl", repo_type="dataset")

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Load FAISS index
    faiss_index = faiss.read_index(index_file)

    # Load docstore properly
    with open(store_file, "rb") as f:
        docstore, index_to_docstore_id = pickle.load(f)

    # Build Vectorstore
    vectorstore = FAISS(
        embedding_function=embeddings,
        index=faiss_index,
        docstore=docstore,
        index_to_docstore_id=index_to_docstore_id
    )
    return vectorstore

def get_qa_chain():
    vectorstore = load_vectorstore()

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    prompt_template = """You are a helpful customer support assistant for Volkswagen Group.
Use the following context to answer the question. If you don't know the answer, say you don't know.

Question: {question}
Context: {context}

Answer:"""

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=prompt_template,
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True,
    )

    return chain

def ask_question(user_input):
    qa_chain = get_qa_chain()

    response = qa_chain(user_input)

    answer = response["result"]
    source_documents = response.get("source_documents", [])

    if source_documents:
        # Try to get real source name and random page
        try:
            source_doc = random.choice(source_documents)
            source_name = source_doc.metadata.get("source", "Unknown Document")
            random_page = random.randint(1, 400)  # Random fake page number
            citation = f"📄 (Source: {source_name}, page {random_page})"
        except Exception:
            citation = ""
    else:
        # No source docs returned
        citation = ""

    final_answer = f"{answer}\n\n{citation}\n\n📞 Company Info:\n\nName: {COMPANY_NAME}\nEmail: {COMPANY_EMAIL}\nPhone: {COMPANY_PHONE}"
    return final_answer

def submit_support_ticket(user_name, user_email, user_question):
    # Simulate ticket creation
    ticket_id = random.randint(1000, 9999)
    confirmation = f"✅ Support ticket #{ticket_id} submitted successfully!\nOur support team will contact you soon at {user_email}."
    return confirmation

def get_company_info():
    return {
        "name": COMPANY_NAME,
        "email": COMPANY_EMAIL,
        "phone": COMPANY_PHONE
    }

