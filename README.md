---
title: "Volkswagen AI Support Chat"
emoji: "💬"
colorFrom: blue
colorTo: green
sdk: streamlit
app_file: app/main.py
pinned: false
---

# Volkswagen AI Support Chat


An AI assistant to answer questions about Volkswagen documents and raise GitHub support tickets when needed.

## 💾 Dataset

This app uses documents and FAISS vector database stored in a Hugging Face dataset:
- **Dataset link**: [Nazokatgmva/volkswagen-support-data](https://huggingface.co/datasets/Nazokatgmva/volkswagen-support-data)

The dataset contains:
- `Y_2024_e.pdf`
- `2023_Volkswagen_Group_Sustainability_Report.pdf`
- `20240930_Group_CoC_Brochure_EN_RGB_V3_1.pdf`
- FAISS index files: `index.faiss` and `index.pkl`

## ✨ Features

- AI assistant answers questions using company documents.
- If no answer found, users are offered to submit a support ticket.
- Support ticket collects: name, email, title, description.
- Agent **cites document name and page number**.
- Fake citations generated randomly if needed.
- Conversation history shown to the user.
- Company information (name, email, phone) always displayed.

## 🛠 Tech Stack

- Python 3.12
- Streamlit
- HuggingFace Hub
- LangChain
- Sentence Transformers
- FAISS
- Hugging Face Datasets

## 🚀 Deployment

This application is deployed on [Hugging Face Spaces](https://huggingface.co/spaces/Nazokatgmva/volkswagen-support-ai).

## 📂 Structure
├── app/
│   ├── main.py
│   └── qa_engine.py
├── data/
│   ├── (PDF files for local testing)
├── faiss_index/
│   ├── index.faiss
│   └── index.pkl
├── rebuild_faiss.py
├── requirements.txt
├── Dockerfile
├── README.md
## 👩‍💻 Author

- Built by Nazokat Gayimova

---

