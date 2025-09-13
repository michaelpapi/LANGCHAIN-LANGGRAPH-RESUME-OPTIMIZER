# Langchain\Langgraph-Resume-O

https://github.com/user-attachments/assets/e39ab94c-d1d5-4abb-80e2-ba369ef2d900

ptimizer

📄 **Resume Optimizer** powered by LangChain & LangGraph with Streamlit UI

***

## Overview

This application automates resume optimization by analyzing uploaded resumes against job descriptions and generating tailored improvement suggestions powered by large language models and vector retrieval techniques.

- **Resume analysis**: Extract key skills, experience, education, projects, career progression from resumes.
- **Optimization suggestions**: Provide actionable improvement advice tailored to the job description.
- **Interactive Streamlit UI**: Upload PDF resumes, enter job info, run optimization, view structured results.

***

## Features

- Resume ingestion & embedding with FAISS vector store
- Retrieval augmented generation (RAG) using LangChain + LangGraph
- Custom output parsing with Pydantic schemas for structured JSON
- Clean Streamlit user interface with sidebar controls & summaries
- Resume re-analysis and iterative refinement

***

## Requirements

- Python 3.10+
- Streamlit 1.47.1+
- langchain-together, langchain-core, langchain-community
- faiss-cpu
- pydantic v2.x
- dotenv
- Other utilities: shutil, tempfile, regex

***

## Setup & Installation

1. Clone the repo:

```bash
git clone <your_repo_url>
cd langgraph-resume-optimizer
```

2. Create and activate a Python virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

3. Install Python dependencies:

```bash
pip install -r requirements.txt
```

4. Create a `.env` file with your API keys (e.g., LLM provider keys).

5. Run the app:

```bash
streamlit run app.py
```

6. In the browser, upload your resume (PDF), input job title and description, select optimization type, and run.

***

## File Structure

```
/
├── app.py                 # Main Streamlit UI and orchestration logic
├── nodes.py               # LangGraph node implementations for embedding, analysis, suggestions
├── graph.py               # LangGraph graph creation and node linkage
├── utils.py               # Utility functions: PDF loading, display, prompt templates
├── requirements.txt       # Python dependencies
├── .env                   # Environment variables (API keys)
└── README.md              # This documentation
```

***

## requirements.txt

```text
streamlit==1.47.1
langchain-together
langchain-core
langchain-community
faiss-cpu
pydantic>=2.0.0
python-dotenv
```

***

## Notes

- Make sure your vector store (FAISS) and LLM API keys are properly configured.
- The app uses Pydantic v2's `.model_dump()` for safe JSON serialization.
- Prompts include strict format instructions to ensure JSON output without extraneous text.
- Output parsing strips internal `<think>...</think>` tags to avoid JSON parse errors.
- Re-analyze triggers Streamlit `st.rerun()` for UI refresh (Streamlit 1.47 compatibility).
- Adjust model names and temperature in `.env` or sidebar to experiment with different LLMs.

***

## Contributing

Contributions, bug reports, or feature requests are welcome! Please open an issue or submit a pull request.

***

## License

MIT License.










Uploading resume-optimizer.mp4…

