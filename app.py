import streamlit as st
import shutil, tempfile, re
import json
from utils import display_pdf_preview, load_resume_documents, OPTIMIZATION_PROMPTS
from graph import create_resume_graph



def format_list(title, items):
    if not items:
        return f"### {title}\nNo data available."
    bullets = "\n".join(f"- {item}" for item in items)
    return f"### {title}\n{bullets}"


def main():
    st.set_page_config(page_title="Resume Optimizer", layout="wide")
    st.title("📄 Resume Optimizer")
    st.caption("Powered by LangChain + LangGraph")

    session_defaults = {
        "messages": [],
        "resume_analysis": {},
        "optimization_suggestions": {},
        "trigger_reanalyze": False,
        "current_pdf": None,
        "documents": None,
        "temp_dir": None,
        "embedding_model": "intfloat/multilingual-e5-large-instruct",
        "generative_model": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free",
        "last_result": None,
    }
    for key, default in session_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default

    with st.sidebar:
        st.image("./img/Nebius.png", width=150)
        st.session_state.generative_model = st.selectbox(
            "Generative Model",
            ["deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free", "gpt-3.5-turbo", "google/gemma-3n-E4B-it"],
            index=0,
        )
        st.session_state.embedding_model = st.selectbox(
            "Embedding Model",
            ["intfloat/multilingual-e5-large-instruct", "sentence-transformers/all-mpnet-base-v2"],
            index=0,
        )
        uploaded_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])
        if uploaded_file and uploaded_file != st.session_state.current_pdf:
            st.session_state.current_pdf = uploaded_file
            if st.session_state.temp_dir:
                shutil.rmtree(st.session_state.temp_dir, ignore_errors=True)
            st.session_state.temp_dir = tempfile.mkdtemp()
            temp_file_path = f"{st.session_state.temp_dir}/{uploaded_file.name}"
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            try:
                st.session_state.documents = load_resume_documents(temp_file_path)
                st.success("Resume loaded successfully!")
                display_pdf_preview(uploaded_file)
            except Exception as e:
                st.error(f"Failed to load PDF: {e}")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Job Information")
        job_title = st.text_input("Job Title", st.session_state.get("job_title", ""))
        job_description = st.text_area("Job Description", st.session_state.get("job_description", ""))
        optimization_type = st.selectbox("Optimization Type", list(OPTIMIZATION_PROMPTS.keys()), index=0)

        if st.button("Optimize Resume"):
            if not st.session_state.documents:
                st.error("Please upload a resume PDF first.")
                st.stop()
            if not job_title or not job_description:
                st.error("Please provide job title and description.")
                st.stop()

            st.session_state["job_title"] = job_title
            st.session_state["job_description"] = job_description
            st.session_state["optimization_type"] = optimization_type

            graph = create_resume_graph()
            graph_state = {
                "documents": st.session_state.documents,
                "embedding_model": st.session_state.embedding_model,
                "generative_model": st.session_state.generative_model,
                "job_title": job_title,
                "job_description": job_description,
                "optimization_query": OPTIMIZATION_PROMPTS[optimization_type],
            }
            try:
                with st.spinner("Running resume optimization..."):
                    result = graph.invoke(graph_state)

                st.session_state.last_result = result

                
                # Convert Pydantic models to dict before storing in session state
                resume_analysis = result.get("resume_analysis", {})
                if hasattr(resume_analysis, "model_dump"):
                    resume_analysis = resume_analysis.model_dump()
                st.session_state.resume_analysis = resume_analysis

                optimization_suggestions = result.get("optimization_suggestions", {})
                if hasattr(optimization_suggestions, "model_dump"):
                    optimization_suggestions = optimization_suggestions.model_dump()
                st.session_state.optimization_suggestions = optimization_suggestions

                analysis_md = (
                    format_list("Key Skills", st.session_state.resume_analysis.get("key_skills"))
                    + "\n\n"
                    + format_list("Professional Experience", st.session_state.resume_analysis.get("professional_experience"))
                    + "\n\n"
                    + format_list("Education", st.session_state.resume_analysis.get("education"))
                    + "\n\n"
                    + format_list("Notable Projects", st.session_state.resume_analysis.get("notable_projects"))
                    + "\n\n"
                    + f"### Career Progression\n{st.session_state.resume_analysis.get('career_progression', 'No data')}"
                )
                suggestions_md = (
                    format_list("Key Findings", st.session_state.optimization_suggestions.get("key_findings"))
                    + "\n\n"
                    + format_list("Specific Improvements", st.session_state.optimization_suggestions.get("specific_improvements"))
                    + "\n\n"
                    + format_list("Action Items", st.session_state.optimization_suggestions.get("action_items"))
                )

                st.session_state.messages = [
                    {"role": "assistant", "content": analysis_md},
                    {"role": "assistant", "content": suggestions_md},
                ]

                st.success("Resume optimized successfully!")
            except Exception as e:
                st.error(f"Error running graph: {e}")

        if st.button("Clear History"):
            st.session_state.messages = []
            st.session_state.resume_analysis = {}
            st.session_state.optimization_suggestions = {}
            st.success("History cleared.")

    with col2:
        st.subheader("Optimization Results")
        if st.session_state.messages:
            for msg in st.session_state.messages:
                st.markdown(msg["content"])
            with st.expander("Raw JSON Output"):
                st.json(st.session_state.last_result or {})
        else:
            st.info("Upload and optimize to see results here.")

        if st.session_state.messages and st.button("🔄 Re-analyze Resume"):
            st.session_state.trigger_reanalyze = True
            st.rerun()

    # Handle re-analyze
    if st.session_state.get("trigger_reanalyze", False):
        st.session_state.trigger_reanalyze = False
        graph = create_resume_graph()
        graph_state = {
            "documents": st.session_state.documents,
            "embedding_model": st.session_state.embedding_model,
            "generative_model": st.session_state.generative_model,
            "job_title": st.session_state.get("job_title", ""),
            "job_description": st.session_state.get("job_description", ""),
            "optimization_query": OPTIMIZATION_PROMPTS.get(st.session_state.get("optimization_type", ""), ""),
        }
        try:
            with st.spinner("Running re-analysis..."):
                result = graph.invoke(graph_state)
            st.session_state.last_result = result
            # Convert Pydantic models to dict before storing in session state
            resume_analysis = result.get("resume_analysis", {})
            if hasattr(resume_analysis, "model_dump"):
                resume_analysis = resume_analysis.model_dump()
            st.session_state.resume_analysis = resume_analysis

            optimization_suggestions = result.get("optimization_suggestions", {})
            if hasattr(optimization_suggestions, "model_dump"):
                optimization_suggestions = optimization_suggestions.model_dump()
            st.session_state.optimization_suggestions = optimization_suggestions
            
            analysis_md = (
                "### Re-Analysis\n"
                + format_list("Key Skills", st.session_state.resume_analysis.get("key_skills"))
                + "\n\n"
                + format_list("Professional Experience", st.session_state.resume_analysis.get("professional_experience"))
                + "\n\n"
                + format_list("Education", st.session_state.resume_analysis.get("education"))
                + "\n\n"
                + format_list("Notable Projects", st.session_state.resume_analysis.get("notable_projects"))
                + "\n\n"
                + f"### Career Progression\n{st.session_state.resume_analysis.get('career_progression', 'No data')}"
            )
            suggestions_md = (
                "### Re-Optimization Suggestions\n"
                + format_list("Key Findings", st.session_state.optimization_suggestions.get("key_findings"))
                + "\n\n"
                + format_list("Specific Improvements", st.session_state.optimization_suggestions.get("specific_improvements"))
                + "\n\n"
                + format_list("Action Items", st.session_state.optimization_suggestions.get("action_items"))
            )

            st.session_state.messages.append({"role": "assistant", "content": analysis_md})
            st.session_state.messages.append({"role": "assistant", "content": suggestions_md})

            st.rerun()
        except Exception as e:
            st.error(f"Error during re-analysis: {e}")


if __name__ == "__main__":
    main()
