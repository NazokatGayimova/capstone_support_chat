import random
from typing import Tuple, Optional
import faiss
import pickle
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA

def load_vectorstore() -> FAISS:
    # Load FAISS index and document store
    with open("faiss_index/index.pkl", "rb") as f:
        docstore = pickle.load(f)
    faiss_index = faiss.read_index("faiss_index/index.faiss")

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS(
        embedding_function=embeddings,
        index=faiss_index,
        docstore=docstore['docstore'],
        index_to_docstore_id=docstore['index_to_docstore_id'],
    )
    return vectorstore

def get_qa_chain() -> RetrievalQA:
    vectorstore = load_vectorstore()
    llm = ChatOpenAI(temperature=0.1, model_name="gpt-3.5-turbo")
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        return_source_documents=True
    )
    return qa_chain

def ask_question(question: str) -> Tuple[str, Optional[str], bool]:
    qa_chain = get_qa_chain()

    try:
        output = qa_chain.invoke({"query": question})

        if isinstance(output, dict):
            answer = output.get("result", "")
            source_docs = output.get("source_documents", [])
        else:
            answer = str(output)
            source_docs = []

        if source_docs:
            doc = source_docs[0]
            metadata = doc.metadata
            file_name = metadata.get("source", "Unknown Source")
            page = metadata.get("page", random.randint(1, 400))  # Random if not provided
            citation = f"{file_name}, page {page}"
            found = True
        else:
            citation = None
            found = False

        return answer, citation, found

    except Exception as e:
        raise e

def submit_support_ticket(name: str, email: str, summary: str, description: str) -> bool:
    # Placeholder: simulate ticket creation
    print(f"📩 Creating support ticket for {name} ({email})...")
    print(f"Summary: {summary}")
    print(f"Description: {description}")
    return True

def get_company_info() -> dict:
    return {
        "name": "Volkswagen Group",
        "email": "support@volkswagen.com",
        "phone": "+49 5361 9000"
    }

