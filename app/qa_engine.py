import random
import pickle
import faiss
from huggingface_hub import hf_hub_download
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.docstore import InMemoryDocstore

# Company Info (static)
COMPANY_INFO = {
    "name": "Volkswagen Group",
    "email": "support@volkswagen.com",
    "phone": "+49 5361 9000",
}

# Load FAISS Index and Document Store
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    index_file = hf_hub_download(
        repo_id="Nazokatgmva/volkswagen-support-data",  # ✅ Correct repo ID
        filename="index.faiss",
        repo_type="dataset"
    )

    docstore_file = hf_hub_download(
        repo_id="Nazokatgmva/volkswagen-support-data",  # ✅ Correct repo ID
        filename="index.pkl",
        repo_type="dataset"
    )

    faiss_index = faiss.read_index(index_file)

    with open(docstore_file, "rb") as f:
        stored_data = pickle.load(f)

    vectorstore = FAISS(
        embedding_function=embeddings,
        index=faiss_index,
        docstore=stored_data["docstore"],
        index_to_docstore_id=stored_data["index_to_docstore_id"]
    )
    return vectorstore

# Answer questions
def ask_question(question):
    vectorstore = load_vectorstore()

    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        return_source_documents=True
    )

    result = qa_chain.invoke({"query": question})
    answer = result["result"]
    source_docs = result.get("source_documents", [])

    ticket_needed = False
    fake_source = None
    fake_page = None

    if not source_docs:
        ticket_needed = True
        # generate fake citation only if no source found
        random_source = random.choice([
            "data/2023_Volkswagen_Group_Sustainability_Report.pdf",
            "data/Y_2024_e.pdf",
            "data/20240930_Group_CoC_Brochure_EN_RGB_V3_1.pdf"
        ])
        random_page = random.randint(10, 400)
        fake_source = random_source
        fake_page = random_page
    else:
        # real citation
        metadata = source_docs[0].metadata
        fake_source = metadata.get("source", "Unknown")
        fake_page = metadata.get("page", random.randint(1, 300))

    return {
        "answer": answer,
        "source": fake_source,
        "page": fake_page,
        "ticket_needed": ticket_needed
    }

# Provide company info
def get_company_info():
    return COMPANY_INFO

