from app.qa_engine import load_documents, create_vectorstore, save_vectorstore

print("🔄 Loading documents...")
docs = load_documents()
print(f"✅ Loaded {len(docs)} pages.")

print("⚙️ Creating vectorstore...")
vectorstore = create_vectorstore(docs)

print("💾 Saving vectorstore to 'faiss_index'...")
save_vectorstore(vectorstore)

print("🎉 Vector index created successfully!")

