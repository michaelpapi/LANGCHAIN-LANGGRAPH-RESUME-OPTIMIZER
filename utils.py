import base64
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader


def display_pdf_preview(pdf_file):
    """Show uploaded PDF in Streamlit sidebar"""
    try:
        st.sidebar.subheader("Resume Preview")
        base64_pdf = base64.b64encode(pdf_file.getvalue()).decode("utf-8")
        iframe = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600px" type="application/pdf"></iframe>'
        st.sidebar.markdown(iframe, unsafe_allow_html=True)
    except Exception as e:
        st.sidebar.error(f"Error previewing PDF: {str(e)}")


def load_resume_documents(pdf_path):
    """Convert uploaded PDF into LangChain documents safely"""
    try:
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        return documents
    except Exception as e:
        st.error(f"Failed to load document: {str(e)}")
        return []

# Map optimization types to specific prompt requests
OPTIMIZATION_PROMPTS = {
    "ATS Keyword Optimizer": "Identify and optimize ATS keywords. Focus on exact matches and semantic variations from the job description.",
    "Experience Section Enhancer": "Enhance experience section to align with job requirements. Focus on quantifiable achievements.",
    "Skills Hierarchy Creator": "Organize skills based on job requirements. Identify gaps and development opportunities.",
    "Professional Summary Crafter": "Create a targeted professional summary highlighting relevant experience and skills.",
    "Education Optimizer": "Optimize education section to emphasize relevant qualifications for this position.",
    "Technical Skills Showcase": "Organize technical skills based on job requirements. Highlight key competencies.",
    "Career Gap Framing": "Address career gaps professionally. Focus on growth and relevant experience.",
}

