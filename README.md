# StartUpLex: Agentic AI Legal Assistant for Startup Founders

StartUpLex is a full-stack Agentic AI platform designed to democratize legal accessibility for early-stage startup founders. Moving beyond basic chatbots, it utilizes a multi-tool autonomous reasoning engine to perform complex legal triage, document generation, and contract analysis.

## 🚀 Key Features

*   ⚖️ **Intelligent Contract Review:** Upload contracts for autonomous red-flag analysis against standard startup terms.
*   📝 **Automated Document Drafting:** Generate customized NDAs, agreements, and founding documents dynamically.
*   ✅ **Compliance Checklists:** Generate step-by-step operational and legal checklists (e.g., hiring contractors, incorporating).
*   🔍 **RAG Search:** Semantic search over local legal databases using FAISS and HuggingFace Embeddings with strict similarity threshold filtering to prevent hallucinations.
*   🌐 **Live Web Search:** Real-time state filing fees and compliance updates via DuckDuckGo integration.
*   🔐 **Secure User Authentication:** Full-stack architecture featuring JWT token-based authentication and SQLite for persistent, secure chat history.
*   💻 **Responsive UI:** Modern HTML5/CSS3/Vanilla JS frontend with real-time Markdown rendering (`marked.js`).

## 🛠️ Tech Stack

*   **Backend:** Python, Flask, SQLite, JWT (JSON Web Tokens)
*   **Frontend:** HTML5, CSS3, Vanilla JavaScript (ES6+), marked.js
*   **AI Framework:** LangChain (AgentExecutor, Tool Calling)
*   **LLM Engine:** Groq API (LLaMA 3 / Open-Source Models)
*   **Retrieval (RAG):** FAISS Vector Store, HuggingFace Embeddings (`all-MiniLM-L6-v2`)
*   **Document Processing:** PyMuPDF (PDF chunking and extraction)

## 🗂️ Project Structure

```
StartUpLex-main/
├── app.py                      # Flask backend, API routes, and JWT Auth
├── db.py                       # SQLite database management for users/chats
├── rag_service.py              # LangChain Agentic AI, Tool Definitions, LLM routing
├── index.html                  # Frontend Landing Page
├── query.html                  # Frontend Chat Interface (Markdown rendered)
├── requirements.txt            # Python dependencies
├── .env                        # Environment variables (API keys, configs)
├── .gitignore                  # Git ignore rules
└── RAG/
    ├── data/                   # Directory to drop raw PDFs for bulk ingestion
    ├── data_ingestion.py       # Script to bulk-ingest PDFs into FAISS
    └── legal_faiss_db/         # FAISS vector database storage
```

## 📦 Installation & Setup

### 1. Prerequisites
*   Python 3.8+
*   A free API key from [Groq](https://console.groq.com)

### 2. Clone & Install
```bash
git clone <repository-url>
cd StartUpLex-main
pip install -r requirements.txt
```

### 3. Configure Environment Variables
Create or open the `.env` file in the root directory and add your Groq API key:
```env
# Flask Configuration
SECRET_KEY=your-secure-secret-key
FLASK_ENV=production

# API Keys
GROQ_API_KEY=your_groq_api_key_here
```

### 4. Run the Application
Start the Flask backend server:
```bash
python app.py
```
*Note: Once the server is running on `http://127.0.0.1:5000`, open your File Explorer and double-click `index.html` to launch the frontend!*

## 📚 Bulk Document Ingestion (Optional)
If you want to manually ingest a large corpus of legal PDFs into your local database rather than using the Web UI upload:
1. Place all your `.pdf` files inside the `RAG/data/` folder.
2. Open your terminal in the root folder and run:
```bash
python RAG/data_ingestion.py
```
This will automatically extract, chunk, embed, and save your documents into the FAISS vector store.

---
*⚠️ **Legal Disclaimer:** This system is for educational and triage purposes only and does not constitute official legal advice. Always consult with a licensed attorney for specific legal matters.*
