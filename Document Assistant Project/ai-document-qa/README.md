# AI Document QA — Retrieval-Augmented Generation (RAG)

A document question-answering system built with **Python, LangChain, ChromaDB, Hugging Face embeddings, and Ollama**.

The system ingests a PDF document, splits it into searchable chunks, converts the chunks into vector embeddings, retrieves the most relevant content for a user's question, and generates an answer using a local large language model.

## Project Overview

This project demonstrates the core architecture of a **Retrieval-Augmented Generation (RAG)** application.

Instead of relying entirely on the language model's internal knowledge, the application retrieves relevant information from a user-provided document and provides that information as context to the model.

### Architecture

```text
                PDF Document
                     │
                     ▼
              PDF Text Extraction
                     │
                     ▼
               Text Chunking
                     │
                     ▼
             Sentence Embeddings
                     │
                     ▼
                 ChromaDB
              Vector Database
                     │
              User Question
                     │
                     ▼
               Similarity Search
                     │
                     ▼
             Relevant Context
                     │
                     ▼
              Ollama / Qwen
                     │
                     ▼
                Final Answer
```

## Technologies Used

* **Python**
* **LangChain**
* **ChromaDB**
* **Hugging Face Sentence Transformers**
* **`all-MiniLM-L6-v2`** embeddings
* **Ollama**
* **Qwen 2.5 0.5B**
* **PyPDF**

## Key Features

* PDF document ingestion
* Text extraction from PDF files
* Recursive text chunking
* Semantic vector embeddings
* Persistent ChromaDB vector storage
* Similarity-based document retrieval
* Local LLM inference through Ollama
* Context-grounded question answering
* Fallback response when information is not available in the document

## How It Works

### 1. Document Ingestion

The PDF is loaded using PyPDF and its text is extracted.

```text
PDF → Extracted Text
```

### 2. Text Chunking

The extracted text is divided into smaller overlapping chunks using LangChain's `RecursiveCharacterTextSplitter`.

This allows individual sections of a large document to be retrieved efficiently.

### 3. Embedding Generation

Each chunk is converted into a numerical vector using:

```text
sentence-transformers/all-MiniLM-L6-v2
```

These embeddings represent the semantic meaning of the text.

### 4. Vector Storage

The embeddings and associated document metadata are stored in **ChromaDB**.

Each chunk receives a deterministic identifier and metadata describing its source and chunk ID.

### 5. Retrieval

When a user asks a question, the question is converted into an embedding and compared against the stored document embeddings.

The most relevant chunks are retrieved using similarity search.

### 6. Generation

The retrieved chunks are provided to a local **Qwen 2.5 0.5B** model running through Ollama.

The model is instructed to answer using the retrieved document context.

If the required information cannot be found in the document, the application is instructed to respond:

```text
I don't know based on the document.
```

## Example

### Question

```text
What is a computer?
```

### Retrieved Context

The system retrieves the relevant section from the computer fundamentals document.

### Answer

```text
A computer is an electronic device, operating under the control
of instructions stored in its own memory that can accept data,
process the data according to specified rules, produce information,
and store the information for future use.
```

The answer is generated using the retrieved document context rather than requiring an external LLM API.

## Project Structure

```text
ai-document-qa/
│
├── app/
│   ├── build_knowledge_base.py
│   ├── chunker.py
│   ├── embeddings.py
│   ├── pdf_loader.py
│   ├── rag.py
│   ├── rag2.py
│   ├── search.py
│   ├── test_embeddings.py
│   ├── test_llm.py
│   ├── test_pdf.py
│   └── vector_store.py
│
├── data/
│   └── computer.pdf
│
├── .env
├── .gitignore
├── requirements.txt
└── README.md
```

> The local ChromaDB database is generated during development and should not be committed to Git.

## Installation

Clone the repository:

```bash
git clone <your-repository-url>
cd ai-document-qa
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Ollama Setup

Install Ollama and make sure it is running.

Pull the model:

```bash
ollama pull qwen2.5:0.5b
```

Verify the model is available:

```bash
ollama list
```

## Build the Knowledge Base

Run the knowledge-base creation script:

```bash
python app/build_knowledge_base.py
```

This extracts the PDF text, creates chunks, generates embeddings, and stores them in ChromaDB.

## Run the RAG Application

```bash
python app/rag.py
```

Enter a question about the document when prompted.

Example:

```text
Ask a question: What is computer hardware?
```

## Configuration

The project uses a local Ollama model for generation, so an OpenAI API key is not required for the current RAG pipeline.

Sensitive environment variables should be stored in `.env` and should never be committed to Git.

## Limitations

This is an MVP implementation designed to demonstrate the fundamental RAG pipeline.

Current limitations include:

* PDF extraction depends on the quality and structure of the source PDF.
* Text embedded inside images or diagrams may not be extracted by the current PDF loader.
* Retrieval currently uses basic similarity search.
* There is no advanced reranking stage.
* There is no web interface yet.
* Evaluation is currently based on manual question-answer testing.

These limitations provide opportunities for future improvements.

## Future Improvements

Potential future versions could include:

* OCR and multimodal document processing
* Better document structure detection
* Retrieval reranking
* Relevance-score thresholds
* Source/page citations in answers
* Automated RAG evaluation
* FastAPI backend
* Web-based chat interface
* Support for multiple documents
* Document upload functionality
* Conversation history
* Docker deployment

## What I Learned

This project was built to understand the complete RAG pipeline rather than simply calling an existing RAG framework.

Key concepts explored:

* PDF document processing
* Text chunking
* Embeddings
* Vector databases
* Semantic similarity search
* Metadata and deterministic document IDs
* Retrieval quality
* Prompt-based grounding
* Local LLM inference
* Debugging and evaluating RAG systems

## Status

**MVP Complete**

The core document ingestion, embedding, retrieval, and generation pipeline is functional.

The project is intentionally kept as a relatively simple RAG implementation before adding more advanced retrieval and document-processing techniques.
