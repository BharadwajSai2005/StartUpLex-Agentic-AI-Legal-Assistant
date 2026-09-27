"""
RAG Service Layer for StartupLex.
Wraps FAISS vector search, HuggingFace embeddings, and Groq LLM
into reusable functions for querying and document ingestion.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# Module-level singleton for lazy-loaded RAG components
_rag_components = None


def get_rag_components():
    """
    Lazily initialize and cache the RAG pipeline components:
    embeddings, vectorstore, and LLM.
    Returns a dict with keys: embeddings, vectorstore, llm.
    """
    global _rag_components
    if _rag_components is not None:
        return _rag_components

    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain_groq import ChatGroq

    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    RAG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'RAG')

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.load_local(
        os.path.join(RAG_DIR, "legal_faiss_db"),
        embeddings,
        allow_dangerous_deserialization=True
    )

    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model="openai/gpt-oss-20b",
        temperature=0,
        max_tokens=2048
    )

    _rag_components = {
        'embeddings': embeddings,
        'vectorstore': vectorstore,
        'llm': llm,
        'rag_dir': RAG_DIR
    }
    return _rag_components


def is_rag_available():
    """Check if RAG components can be loaded. Returns True/False."""
    try:
        get_rag_components()
        return True
    except Exception:
        return False


# ──────────────────────────────── Similarity Threshold ────────────────────────────────

SIMILARITY_THRESHOLD = 2.3


# ──────────────────────────────── Internal Helpers ────────────────────────────────

def _build_rewrite_chain(llm):
    """Build the adaptive query rewriting chain."""
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    rewrite_prompt = ChatPromptTemplate.from_template("""
You are preparing a search query for legal documents.

Rules:
- Expand acronyms and vague phrases into formal legal language (e.g., "LLC" -> "Limited Liability Company").
- For unfamiliar terms, keep them as-is or use common legal equivalents.
- Do NOT invent laws, cases, or regulations that don't exist.
- Do NOT answer the question itself.
- Output ONLY the rewritten search query, nothing else.

User question:
{question}
""")
    return rewrite_prompt | llm | StrOutputParser()


def _build_k_chain(llm):
    """Build the dynamic-k decision chain."""
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    k_prompt = ChatPromptTemplate.from_template("""
Decide how many legal document chunks are needed to answer this question.

Guidelines:
- Simple factual questions (e.g., "What is statute of limitations?") -> 2-3 chunks
- Moderate complexity (e.g., "What are requirements for forming an LLC?") -> 5-7 chunks
- High complexity (e.g., "What are exceptions to contract enforceability?") -> 10-15 chunks
- Multi-part questions or rare topics -> 15-20 chunks

Question:
{question}

Return ONLY a single integer between 1 and 20.
""")
    return k_prompt | llm | StrOutputParser()


def _retrieve_context(user_question, vectorstore, llm):
    """
    Run adaptive query rewriting, dynamic-k selection, and similarity search.
    Returns the combined context string, or None if nothing relevant was found.
    """
    # Rewrite short/vague queries
    if len(user_question.split()) <= 4:
        rewrite_chain = _build_rewrite_chain(llm)
        retrieval_query = rewrite_chain.invoke({"question": user_question}).strip()
    else:
        retrieval_query = user_question

    # Decide how many chunks to retrieve
    k_chain = _build_k_chain(llm)
    try:
        k = int(k_chain.invoke({"question": retrieval_query}).strip())
    except ValueError:
        k = 5
    k = max(1, min(k, 20))

    # Retrieve similar chunks
    results = vectorstore.similarity_search_with_score(retrieval_query, k=k)

    if not results:
        return None

    # Filter by similarity threshold
    scores = [score for _, score in results]
    if min(scores) > SIMILARITY_THRESHOLD:
        return None

    return "\n\n".join(doc.page_content for doc, _ in results)


# ──────────────────────────────── Public API ────────────────────────────────

def run_query(user_question: str) -> str:
    """
    Single-turn RAG query (backward-compatible with the original query.py).
    Returns the answer string.
    """
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnablePassthrough

    components = get_rag_components()
    vectorstore = components['vectorstore']
    llm = components['llm']

    context = _retrieve_context(user_question, vectorstore, llm)
    if context is None:
        return "I don't know"

    answer_prompt = ChatPromptTemplate.from_template("""
You are a legal assistant answering based solely on provided documents.

IMPORTANT: Answer in plain paragraph format, not numbered lists.

Rules:
- Use ONLY information from the Context below
- If the Context does NOT contain relevant information, respond with: "I don't know"
- Do NOT use external knowledge or make assumptions
- Do NOT cite laws or cases not mentioned in the Context
- If the answer is partial or incomplete, still provide what is available
- Format your answer as regular prose/paragraphs

Context:
{context}

Question:
{question}
""")

    rag_chain = (
        {"context": lambda _: context, "question": RunnablePassthrough()}
        | answer_prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain.invoke(user_question)


def run_chat_query(user_question: str, chat_history: list) -> str:
    """
    Multi-turn Agentic RAG query with conversation history and tools.
    chat_history is a list of dicts: [{"role": "user"/"assistant", "content": "..."}]
    Returns the answer string.
    """
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain.agents import create_tool_calling_agent, AgentExecutor
    from langchain_core.messages import HumanMessage, AIMessage
    from langchain_core.tools import tool

    components = get_rag_components()
    vectorstore = components['vectorstore']
    llm = components['llm']

    # ──────────────────────────────── Tools ────────────────────────────────
    @tool
    def search_local_legal_database(query: str) -> str:
        """Search the local FAISS database for legal documents, standard templates, or uploaded contracts."""
        context = _retrieve_context(query, vectorstore, llm)
        if not context:
            return "No relevant local documents found in the database."
        return context

    @tool
    def search_live_web(query: str) -> str:
        """Search the live internet for up-to-date filing fees, state-specific rules, or recent case law."""
        try:
            from duckduckgo_search import DDGS
            results = DDGS().text(query, max_results=3)
            if not results:
                return "No web results found."
            return "\n".join([f"- {r['title']}: {r['body']}" for r in results])
        except Exception as e:
            return f"Web search failed: {e}"

    @tool
    def draft_legal_document(document_type: str, details: str) -> str:
        """
        Drafts custom legal documents (e.g., NDAs, Founder Agreements, Contracts).
        Provide the document_type and specific details (names, dates, clauses).
        """
        prompt = ChatPromptTemplate.from_template(
            "You are an expert startup lawyer. Draft a formal {document_type} based on the following details: {details}"
        )
        chain = prompt | llm
        response = chain.invoke({"document_type": document_type, "details": details})
        return f"--- DRAFT {document_type.upper()} ---\n\n{response.content}"

    @tool
    def review_uploaded_contract(contract_text: str) -> str:
        """
        Reviews a contract or term sheet to find legal red flags or missing standard terms.
        Pass the text of the contract to be reviewed.
        """
        prompt = ChatPromptTemplate.from_template(
            "You are an expert startup lawyer. Review the following contract for legal red flags, unusual terms, or missing standard startup clauses:\n\n{contract_text}"
        )
        chain = prompt | llm
        response = chain.invoke({"contract_text": contract_text})
        return f"--- CONTRACT REVIEW ---\n\n{response.content}"

    @tool
    def generate_step_by_step_checklist(task: str) -> str:
        """Creates actionable step-by-step checklists for complex legal tasks like incorporation or hiring."""
        prompt = ChatPromptTemplate.from_template(
            "Create a step-by-step legal and operational checklist for a startup founder for the following task: {task}"
        )
        chain = prompt | llm
        response = chain.invoke({"task": task})
        return f"--- CHECKLIST: {task.upper()} ---\n\n{response.content}"

    tools = [
        search_local_legal_database,
        search_live_web,
        draft_legal_document,
        review_uploaded_contract,
        generate_step_by_step_checklist
    ]

    # ──────────────────────────────── History & Prompt ────────────────────────────────
    # Build conversation history
    recent_history = chat_history[-10:]  # up to 5 exchanges
    formatted_history = []
    for msg in recent_history:
        if msg['role'] == 'user':
            formatted_history.append(HumanMessage(content=msg['content']))
        else:
            formatted_history.append(AIMessage(content=msg['content']))

    system_prompt = """You are an Agentic Virtual Startup Paralegal.
    
Your goal is to help startup founders with legal queries, document drafting, contract review, and operational checklists.
- INTERACTIVE TRIAGE: If the founder's request is too broad (e.g., "I want to start a company", "How do I hire?"), ASK clarifying questions first (e.g., "Are you raising funding?").
- Use 'search_local_legal_database' for standard legal concepts.
- Use 'search_live_web' for current state filing fees, recent laws, or live info.
- Use 'draft_legal_document' to generate NDAs, founder agreements, etc., if asked.
- Use 'review_uploaded_contract' if the user pastes contract text for review.
- Use 'generate_step_by_step_checklist' when guiding founders through complex processes.

Answer directly, professionally, and in plain paragraph format.
"""

    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    # ──────────────────────────────── Agent Execution ────────────────────────────────
    agent = create_tool_calling_agent(llm, tools, chat_prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    try:
        result = agent_executor.invoke({
            "input": user_question,
            "chat_history": formatted_history
        })
        return result["output"]
    except Exception as e:
        return f"I encountered an error while processing your request: {e}"


def ingest_pdf_bytes(file_bytes: bytes, filename: str) -> dict:
    """
    Ingest a PDF from raw bytes into the FAISS vector store.
    Returns dict with 'chunks_added' and 'filename'.
    """
    import fitz  # PyMuPDF
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_core.documents import Document

    components = get_rag_components()
    vectorstore = components['vectorstore']
    rag_dir = components['rag_dir']

    # Extract text from PDF bytes
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    pages_text = []
    for page in doc:
        text = page.get_text()
        if text.strip():
            pages_text.append(text)
    doc.close()

    if not pages_text:
        return {'chunks_added': 0, 'filename': filename}

    # Create Document objects
    documents = [
        Document(page_content=text, metadata={"source": filename, "page": i})
        for i, text in enumerate(pages_text)
    ]

    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=250,
        chunk_overlap=50
    )
    chunks = text_splitter.split_documents(documents)

    if not chunks:
        return {'chunks_added': 0, 'filename': filename}

    # Add to existing vectorstore and persist
    vectorstore.add_documents(chunks)
    vectorstore.save_local(os.path.join(rag_dir, "legal_faiss_db"))

    return {'chunks_added': len(chunks), 'filename': filename}
