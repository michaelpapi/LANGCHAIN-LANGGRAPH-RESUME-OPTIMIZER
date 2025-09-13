# test_nodes.py

import os
from dotenv import load_dotenv
from nodes import analyze_resume, generate_suggestions
from langchain.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_together import TogetherEmbeddings

load_dotenv()

# Load your sample resume PDF from local path for testing
SAMPLE_PDF_PATH = "./docs/CV2.pdf"

def load_documents(pdf_path):
    loader = PyPDFLoader(pdf_path)
    return loader.load()

def test_analyze_resume():
    print("Loading documents...")
    documents = load_documents(SAMPLE_PDF_PATH)
    print(f"Loaded {len(documents)} documents.")

    embedder = TogetherEmbeddings(model="intfloat/multilingual-e5-large-instruct")
    vectorstore = FAISS.from_documents(documents, embedder)
    print("Documents embedded into vectorstore.")

    state = {
        "vectorstore": vectorstore,
        "generative_model": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free",
    }
    print("Running analyze_resume node...")
    result = analyze_resume(state)
    print("analyze_resume output:")
    print(result.get("resume_analysis", "[No analysis returned]"))

def test_generate_suggestions():
    analysis_text = """
- Strong Python skills highlighted.
- 5+ years experience in machine learning.
- Education: MSc Computer Science.
"""
    state = {
        "resume_analysis": analysis_text,
        "job_title": "Machine Learning Engineer",
        "job_description": "Looking for an engineer with experience in ML pipelines and Python coding.",
        "optimization_query": "Enhance experience section to align with job requirements. Focus on quantifiable achievements.",
        "generative_model": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free"
    }
    print("Running generate_suggestions node...")
    result = generate_suggestions(state)
    print("generate_suggestions output:")
    print(result.get("optimization_suggestions", "[No suggestions returned]"))

if __name__ == "__main__":
    test_analyze_resume()
    print("\n" + "="*40 + "\n")
    test_generate_suggestions()
