import streamlit as st
import json
from typing import List
from pydantic import BaseModel, Field
from langchain_together import ChatTogether, TogetherEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import PydanticOutputParser
from dotenv import load_dotenv
import re

load_dotenv()



def strip_think_tags(text: str) -> str:
    """Remove <think>...</think> blocks from LLM output."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


class ResumeAnalysisSchema(BaseModel):
    key_skills: List[str] = Field(default_factory=list)
    professional_experience: List[str] = Field(default_factory=list)
    education: List[str] = Field(default_factory=list)
    notable_projects: List[str] = Field(default_factory=list)
    career_progression: str = ""

class SuggestionsSchema(BaseModel):
    key_findings: List[str] = Field(default_factory=list)
    specific_improvements: List[str] = Field(default_factory=list)
    action_items: List[str] = Field(default_factory=list)

resume_parser = PydanticOutputParser(pydantic_object=ResumeAnalysisSchema)
suggestions_parser = PydanticOutputParser(pydantic_object=SuggestionsSchema)

def clean_llm_output(text):
    # Strip markdown triple backticks, if any
    cleaned = re.sub(r"``````", "", text, flags=re.IGNORECASE).strip()
    return cleaned

def embed_documents(state):
    try:
        st.write("Running embed_documents node...")
        documents = state.get("documents", [])
        if not documents:
            raise ValueError("No documents to embed")
        embedding_model = state.get("embedding_model", "intfloat/multilingual-e5-large-instruct")
        embedder = TogetherEmbeddings(model=embedding_model)
        vectorstore = FAISS.from_documents(documents, embedder)
        st.write("Embedding complete")
        state["vectorstore"] = vectorstore
        return state
    except Exception as e:
        st.error(f"Error in embed_documents: {e}")
        state["vectorstore"] = None
        return state

def analyze_resume(state):
    try:
        st.write("Running analyze_resume node...")
        vectorstore = state.get("vectorstore")
        if vectorstore is None:
            raise ValueError("Vectorstore missing")

        llm = ChatTogether(
            model=state.get("generative_model", "deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free"),
            temperature=0
        )

        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

        query = state.get("job_description") or state.get("job_title") or "resume"
        st.write(f"Retrieving documents for query: {query[:80]}...")

        context_docs = retriever.invoke(query)
        if hasattr(context_docs, "docs"):
            context_docs = context_docs.docs

        context_text = "\n\n".join(doc.page_content for doc in context_docs)

        prompt = f"""
Please provide a JSON object strictly matching the schema below.
Do not include any explanations or other text.

{resume_parser.get_format_instructions()}

Resume Content:
{context_text}
"""

        response = llm.invoke([{"role": "user", "content": prompt}])
        raw_text = clean_llm_output(response.content)

        try:
            parsed = resume_parser.parse(raw_text)
        except Exception as e:
            st.error(f"Failed to parse resume analysis output: {e}")
            parsed = ResumeAnalysisSchema().model_dump()

        state["resume_analysis"] = parsed
        st.write("Got structured analysis response")
        return state
    except Exception as e:
        st.error(f"Error in analyze_resume: {e}")
        state["resume_analysis"] = {}
        return state

def generate_suggestions(state):
    try:
        st.write("Running generate_suggestions node...")

        llm = ChatTogether(
            model=state.get("generative_model", "deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free"),
            temperature=0
        )

        resume_analysis_obj = state.get("resume_analysis", ResumeAnalysisSchema())
        if hasattr(resume_analysis_obj, "model_dump"):
            analysis_dict = resume_analysis_obj.model_dump()
        else:
            analysis_dict = resume_analysis_obj  # fallback in case it's already dict
        resume_analysis_json = json.dumps(analysis_dict)

        job_title = state.get("job_title", "")
        job_description = state.get("job_description", "")
        optimization_query = state.get("optimization_query", "")

        prompt = f"""
Please provide a JSON object strictly matching the schema below.
Do not include any explanations or other text.

{suggestions_parser.get_format_instructions()}

Resume Analysis:
{resume_analysis_json}

Job Title: {job_title}
Job Description: {job_description}

Optimization Request: {optimization_query}
"""

        response = llm.invoke([{"role": "user", "content": prompt}])
        raw_text = clean_llm_output(response.content)

        try:
            parsed = suggestions_parser.parse(raw_text)
        except Exception as e:
            st.error(f"Failed to parse suggestions output: {e}")
            parsed = SuggestionsSchema().model_dump()

        state["optimization_suggestions"] = parsed
        st.write("Got structured suggestions response")
        return state

    except Exception as e:
        st.error(f"Error in generate_suggestions: {e}")
        state["optimization_suggestions"] = {}
        return state

def check_reanalyze(state):
    try:
        trigger_reanalyze = state.get("trigger_reanalyze", False)
        st.write(f"check_reanalyze node: trigger_reanalyze={trigger_reanalyze}")
        state["reanalyze"] = trigger_reanalyze
        return state
    except Exception as e:
        st.error(f"Error in check_reanalyze: {e}")
        state["reanalyze"] = False
        return state
