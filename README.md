# AI Resume Analyzer

An AI-powered Resume Analysis system that extracts text from PDF resumes, computes ATS-style similarity scores, and generates structured job-fit evaluations using Streamlit, LangChain, Gemini LLM, and BERT embeddings.

## Features
- Upload and parse PDF resumes using pdfminer.six
- ATS-style semantic similarity scoring with SentenceTransformer embeddings
- AI-generated evaluation report using Gemini + LangChain
- Point-wise scoring (out of 5) with ✔/❌/⚠ indicators
- Automated improvement suggestions and downloadable reports
- Clean and interactive Streamlit UI

## Tech Stack
- *Frontend:* Streamlit  
- *LLM & Workflow:* Gemini 2.5 Flash, LangChain  
- *Embeddings:* SentenceTransformers (all-mpnet-base-v2)  
- *PDF Extraction:* pdfminer.six  
- *Similarity Metric:* Cosine Similarity (sklearn)

## Command to Run
```bash
streamlit run <filelocation>
