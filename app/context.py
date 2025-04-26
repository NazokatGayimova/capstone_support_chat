# context.py

from huggingface_hub import hf_hub_download
import os

def get_pdf_path():
    pdf_path = hf_hub_download(
        repo_id="Nazokatgmva/volkswagen-support-data",
        filename="Y_2024_e.pdf",
        repo_type="dataset"
    )
    return pdf_path

def get_faiss_path():
    index_path = hf_hub_download(
        repo_id="Nazokatgmva/volkswagen-support-data",
        filename="index.faiss",
        repo_type="dataset"
    )
    return index_path
