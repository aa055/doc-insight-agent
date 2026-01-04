# Doc Insight Agent

A chat-based Retrieval-Augmented Generation (RAG) application for PDF document analysis. Upload PDFs and interact with your documents using local AI - ask questions, get summaries, or generate structured notes.

## Features

- **Document Q&A**: Ask questions about your uploaded documents with source citations
- **Summarization**: Get concise summaries of individual or all documents
- **Structured Notes**: Generate organized notes with bullet points and key takeaways
- **Multi-PDF Support**: Upload and query across multiple documents
- **Local AI**: Runs entirely on your machine using Ollama

## Tech Stack

- **LangChain**: Document processing and RAG pipeline
- **Ollama**: Local LLM and embeddings
- **FAISS**: Vector storage for document retrieval
- **Gradio**: Web-based chat interface

## Prerequisites

1. **Python 3.10+**

2. **Ollama** - Install from [ollama.com](https://ollama.com)

3. **Required Models** - After installing Ollama, pull the models:
   ```bash
   ollama pull llama3.2
   ollama pull nomic-embed-text
   ```

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd doc-insight-agent
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv

   # Windows
   venv\Scripts\activate

   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment** (optional):
   ```bash
   cp .env.example .env
   # Edit .env to customize settings
   ```

## Usage

1. **Start Ollama** (if not running as a service):
   ```bash
   ollama serve
   ```

2. **Run the application**:
   ```bash
   python -m src.app
   ```

3. **Open your browser** at `http://127.0.0.1:7860`

4. **Upload PDFs** using the file upload section

5. **Select a mode**:
   - **Q&A**: Ask questions about your documents
   - **Summarize**: Get document summaries
   - **Generate Notes**: Create structured notes

6. **Optionally filter** by specific document using the dropdown

## Configuration

Environment variables (set in `.env`):

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `LLM_MODEL` | `llama3.2` | Chat model to use |
| `EMBEDDING_MODEL` | `nomic-embed-text` | Embedding model |
| `CHUNK_SIZE` | `1000` | Text chunk size |
| `CHUNK_OVERLAP` | `200` | Chunk overlap |
| `RETRIEVAL_K` | `4` | Number of chunks to retrieve |

## Project Structure

```
doc-insight-agent/
├── src/
│   ├── app.py                 # Main Gradio application
│   ├── config.py              # Configuration settings
│   ├── core/
│   │   ├── document_processor.py  # PDF loading & splitting
│   │   ├── llm.py                 # Ollama LLM wrapper
│   │   └── vector_store.py        # FAISS vector store & embeddings
│   ├── chains/
│   │   ├── qa_chain.py            # Q&A with retrieval
│   │   ├── summarization_chain.py # Document summarization
│   │   └── notes_chain.py         # Structured notes
│   └── utils/
│       └── prompts.py             # Prompt templates
├── data/
│   ├── uploads/               # Temporary PDF storage
│   └── vector_store/          # FAISS index storage
├── requirements.txt
└── README.md
```

## Alternative Models

You can use different Ollama models by updating `.env`:

**LLM Models**:
- `llama3.2` (default, 3B parameters)
- `llama3.1` (8B parameters, better quality)
- `mistral` (7B parameters)
- `qwen2.5` (various sizes)

**Embedding Models**:
- `nomic-embed-text` (default)
- `mxbai-embed-large` (larger, better quality)
- `all-minilm` (smaller, faster)

## License

MIT
