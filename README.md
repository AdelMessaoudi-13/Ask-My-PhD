# 🎓 Ask My PhD — An AI Agent to Explore My Thesis

<p align="center">
  <img src="https://img.shields.io/badge/license-MIT-green.svg" alt="MIT License"/>
  <a href="https://theses.fr/2023AIXM0306">
    <img src="https://img.shields.io/badge/PhD%20Thesis-Access-blue?logo=academia" alt="PhD Thesis"/>
  </a>
  <a href="https://huggingface.co/spaces/AdelMessaoudi-13/Ask-My-PhD">
    <img src="https://img.shields.io/badge/HuggingFace-Live%20Demo-orange?logo=huggingface" alt="Live Demo on Hugging Face"/>
  </a>
  <a href="https://github.com/AdelMessaoudi-13">
    <img src="https://img.shields.io/badge/GitHub-AdelMessaoudi--13-black?logo=github" alt="GitHub"/>
  </a>
  <a href="https://www.linkedin.com/in/adel-messaoudi-831358132">
    <img src="https://img.shields.io/badge/LinkedIn-Profile-blue?logo=linkedin" alt="LinkedIn"/>
  </a>
</p>

An interactive AI assistant that helps users **explore**, **understand**, and **ask questions** about my PhD thesis using natural language and a Retrieval-Augmented Generation (RAG) system.

---

## 🚀 Overview

This app makes my thesis more accessible by allowing users to:

- Ask questions about methods, chapters, or results
- Get cited answers based on real excerpts from the thesis
- Understand key concepts without reading all 100+ pages

---

## ✨ Features

- 🧠 RAG-based document querying
- 🔍 Semantic search with vector embeddings
- 🧾 OCR extraction using **Mistral OCR**
- 📖 Structured access to chapters and sections
- 🤖 Powered by `pydantic-ai`, `Supabase`, and `OpenAI`

---

## 🛠️ Tech Stack

- `Mistral OCR` (PDF text extraction)
- `pydantic-ai` (agent orchestration)
- `Supabase` + `pgvector` (semantic search backend)
- `OpenAI` (GPT-4o mini + embeddings)
- `Streamlit` (user interface)

---

## ⚙️ How to Run the Project Locally

To deploy and run this RAG agent locally with your own thesis:

### 1. Prerequisites

- Python 3.11+
- A Supabase project with `pgvector` enabled
- OpenAI API key
- Mistral API key

### 2. Environment Setup

```bash
git clone https://github.com/AdelMessaoudi-13/ask-my-phd.git
cd ask-my-phd
pip install -r requirements.txt
```

### 3. Environment Variables

Create a `.env` file from the provided template:

```bash
cp .env.example .env
```

Then fill in the required values in `.env`:

```env
MISTRAL_API_KEY=your_mistral_api_key
OPENAI_API_KEY=your_openai_api_key
SUPABASE_URL=https://your_project.supabase.co
SUPABASE_SERVICE_KEY=your_supabase_service_key
```

### 4. Initialize the Supabase Database

In the Supabase dashboard:

- Go to **SQL Editor**
- Paste the contents of `setup_database.sql`
- Click **Run**

This will:

- Create the `document_chunks` table
- Enable `pgvector` for semantic search
- Define the `match_document_chunks()` function
- Add relevant indexes and RLS policies

### 5. Ingest the Thesis

Run the ingestion script:

```bash
python pdf_rag_processor.py --pdf path/to/your_documents.pdf
```

This will:

- Perform OCR via Mistral
- Split the document into structured chunks
- Generate summaries and titles using GPT
- Store everything in Supabase

### 6. Launch the App

```bash
streamlit run user_interface.py
```

Visit: [http://localhost:8501](http://localhost:8501)

---

## 📚 Source

This project is inspired by the work of [Cole Medin](https://github.com/coleam00/ottomator-agents/tree/main/crawl4AI-agent), which proposes an agent-based pipeline for document processing and indexing.

Key adaptations include:

- Replacing web scraping with **Mistral OCR** for high-quality PDF extraction
- Creating a **chapter-aware ingestion pipeline** tailored to the structure of a PhD thesis
- Implementing **custom metadata**, including chapter tracking and source identification
- Designing new tools and prompts to support **semantic RAG-based exploration** of scientific content
- Reengineering the agent logic to guide users through complex academic material — with support for cross-chapter reasoning, explanation, and synthesis
- Reusing and adapting the core Streamlit UI structure, including conversation history management and streaming agent output

This project remains under the **MIT License**, in accordance with the original.  
The original license from Cole Medin is included in the [`THIRD_PARTY_LICENSES`](./THIRD_PARTY_LICENSES) file.

---

## 👤 Author

**Adel Messaoudi**  
🎓 PhD in Applied Mathematics and Mechanics   
🌐 [Mail](amessaoudi.am@gmail.com)  
