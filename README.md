---
title: 💬 Volkswagen AI Customer Support
emoji: 🤖
colorFrom: indigo
colorTo: lime
sdk: docker
sdk_version: 1.0
app_file: app/main.py
pinned: false
---

# 💬 Volkswagen AI Customer Support

An AI-powered assistant that answers user questions based on official Volkswagen documents — and creates support tickets when needed.

Built for my Generative AI Capstone Project using:
- 🧠 LangChain + OpenAI for document Q&A
- 📚 FAISS for fast PDF search
- 💬 Streamlit interface
- 📩 GitHub support ticket integration
- 🚀 Deployed on Hugging Face Spaces (Docker runtime)

---

## ✨ What Can It Do?

✅ Ask questions about Volkswagen  
✅ Shows document source and page  
✅ Offers to create support ticket if answer is unclear  
✅ Sends GitHub issue with user name, email, and description  
✅ Friendly, support-style chat layout  

---

## 🚀 How to Use

1. Type your question (e.g. *“What electric vehicles does Volkswagen sell?”*)  
2. If AI doesn't know, it offers to create a support ticket  
3. Fill in your name, email, and message  
4. Ticket is sent directly to a GitHub repo 🎫

---

## 🛠️ Tech Stack

- **Python 3.10**
- [Streamlit](https://streamlit.io/)
- [LangChain](https://www.langchain.com/)
- [OpenAI API](https://platform.openai.com/)
- [FAISS](https://github.com/facebookresearch/faiss)
- [GitHub REST API](https://docs.github.com/en/rest)

---

## 🔐 Required Secrets

Set these in the Hugging Face Space **Secrets** panel:

```env
OPENAI_API_KEY=your-openai-key
GITHUB_TOKEN=your-github-token
GITHUB_REPO=username/repo-name
