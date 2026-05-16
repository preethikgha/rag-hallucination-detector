# RAG with Hallucination Detection

A production-style Retrieval-Augmented Generation (RAG) pipeline with an LLM-as-judge faithfulness scoring layer that detects hallucinations in AI-generated answers.

**Average faithfulness score: 0.95/1.0 across test queries**

---

##  What This Project Does

Most RAG systems just retrieve and generate — they don't check if the answer is actually grounded in the source. This project adds a **hallucination detection layer** that scores every answer for faithfulness, identifying unsupported claims sentence by sentence.

---

##  Architecture

```
User Query
    ↓
Query Embedding (sentence-transformers)
    ↓
Vector Retrieval (ChromaDB)
    ↓
Top-K Context Chunks
    ↓
LLM Generation (Groq - LLaMA 3.3 70B)
    ↓
Raw Answer
    ↓                    ↓
Final Output      Faithfulness Scorer (LLM-as-judge)
                         ↓
                  Score (0–1) + Verdict + Unsupported Claims
```

---

## Project Structure

```
rag/
├── reingest.py        # PDF ingestion pipeline (PyMuPDF + ChromaDB)
├── rag.py             # Core RAG retrieval and generation logic
├── hallucination.py   # Hallucination detection and faithfulness scoring
├── api.py             # FastAPI backend serving REST API + HTML frontend
├── static/
│   └── index.html     # Clean HTML/CSS/JS frontend UI
├── docs/
│   └── a-practical-guide-to-building-agents.pdf
└── .gitignore
```

---

##  Results

| Question | Faithfulness Score | Verdict |
|---|---|---|
| What is an agent? | 0.9 | ✅ Faithful |
| What are the three core components? | 1.0 | ✅ Faithful |
| When should you build an agent? | 1.0 | ✅ Faithful |
| What are guardrails in agents? | 0.9 | ✅ Faithful |
| How does the triage agent hand off? | 1.0 | ✅ Faithful |
| **Average** | **0.95** | ✅ |

---

##  Key Engineering Decisions

**1. PyMuPDF over PyPDF**
Initial ingestion used PyPDF which produced garbled text (`a v ailable`, `w orkflo w s`) from the scanned PDF. Switching to PyMuPDF and cleaning whitespace improved retrieval quality significantly, pushing faithfulness scores from 0.6 → 0.95.

**2. LLM-as-judge scoring**
Rather than rule-based NLI checking, used a second LLM call to score faithfulness. Returns a 0–1 score, verdict, reason, and specific unsupported claims.

**3. Strict generation prompt**
Tuned the generation prompt with explicit rules to stay grounded in context, preventing the LLM from adding outside knowledge.

---

##  Tech Stack

- **Python** — core language
- **LangChain** — document loading, chunking, retrieval chain
- **ChromaDB** — local vector database
- **sentence-transformers** — `all-MiniLM-L6-v2` embeddings (free, no API)
- **PyMuPDF** — clean PDF text extraction
- **Groq API** — LLaMA 3.3 70B for generation and scoring (free tier)
- **FastAPI** — REST API backend
- **HTML/CSS/JS** — frontend UI 

---

##  Setup & Run

**1. Clone the repo**
```bash
git clone https://github.com/yourusername/rag-hallucination-detection
cd rag-hallucination-detection
```

**2. Install dependencies**
```bash
pip install langchain langchain-community langchain-text-splitters chromadb sentence-transformers pymupdf groq fastapi uvicorn
```

**3. Add your Groq API key**

Get a free key at [console.groq.com](https://console.groq.com), then paste it in `api.py`:
```python
GROQ_API_KEY = "your_key_here"
```

**4. Ingest the PDF**
```bash
python reingest.py
```

**5. Run the API**
```bash
uvicorn api:app --reload
```

**6. Open the UI**
```
http://localhost:8000
```

---

##  Sample Questions to Try

- What is an agent?
- What are the three core components of an agent?
- What are guardrails in agents?
- How does the triage agent hand off to other agents?
- When should you build an agent?
- What is the churn detection guardrail?

---

##  Author

**Preethikgha M** 

[LinkedIn](https://linkedin.com/in/yourprofile) · [GitHub](https://github.com/yourusername)
