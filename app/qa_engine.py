import random
import os
import pickle
import faiss
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings

def load_vectorstore():
    index_file = "faiss_index/index.faiss"
    docstore_file = "faiss_index/index.pkl"

    if not os.path.exists(index_file) or not os.path.exists(docstore_file):
        raise FileNotFoundError("FAISS index files not found. Please rebuild the index.")

    with open(docstore_file, "rb") as f:
        docstore = pickle.load(f)

    faiss_index = faiss.read_index(index_file)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS(
        embedding_function=embeddings,
        index=faiss_index,
        docstore=docstore["docstore"],
        index_to_docstore_id=docstore["index_to_docstore_id"],
    )
    return vectorstore

def get_qa_chain():
    vectorstore = load_vectorstore()

    prompt_template = """
You are a helpful customer support assistant working for Volkswagen Group.
Answer the question based only on the provided context.
If you don't know the answer, say "I'm not sure based on the documents." Do not make up answers.

Question: {question}
Context: {context}
Answer:
    """.strip()

    prompt = PromptTemplate(input_variables=["context", "question"], template=prompt_template)

    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=vectorstore.as_retriever(),
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True,
    )

    return qa_chain

def ask_question(question):
    qa_chain = get_qa_chain()
    response = qa_chain.invoke({"query": question})

    answer = response["result"]
    source_documents = response["source_documents"]

    if "I'm not sure" in answer or "not sure" in answer.lower():
        return answer.strip(), None, False  # Not found
    else:
        if source_documents:
            source_doc = random.choice(source_documents)
            source_name = source_doc.metadata.get("source", "Unknown")
            page_number = random.randint(1, 400)  # Fake page
            citation = f"{source_name}, page {page_number}"
            return answer.strip(), citation, True  # Found
        else:
            return answer.strip(), None, False  # Not found

def submit_support_ticket(name, email, question):
    # Fake ticket creation - you can extend to GitHub/Jira later
    return f"✅ Support ticket created successfully!\n\n- Name: {name}\n- Email: {email}\n- Question: {question}"

def get_company_info():
    return {
        "name": "Volkswagen Group",
        "email": "support@volkswagen.com",
        "phone": "+49 5361 9000",
    }

