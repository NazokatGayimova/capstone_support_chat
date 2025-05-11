from huggingface_hub import hf_hub_download
import os

REPO_ID = "Nazokatgmva/volkswagen-support-data"

def get_pdf_paths():
    # List all the PDF files in your dataset
    pdf_files = [
        "Y_2024_e.pdf",
        "2023_Volkswagen_Group_Sustainability_Report.pdf",
        "20240930_Group_CoC_Brochure_EN_RGB_V3_1.pdf"
    ]
    # Download each PDF and return the paths
    pdf_paths = [hf_hub_download(repo_id=REPO_ID, filename=pdf, repo_type="dataset") for pdf in pdf_files]
    return pdf_paths

def get_faiss_index_paths():
    # Download FAISS index files
    index_file = hf_hub_download(repo_id=REPO_ID, filename="index.faiss", repo_type="dataset")
    pkl_file = hf_hub_download(repo_id=REPO_ID, filename="index.pkl", repo_type="dataset")
    return index_file, pkl_file
