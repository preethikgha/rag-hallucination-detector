from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from groq import Groq
import time
import os
from dotenv import load_dotenv
load_dotenv()

app = FastAPI(title="RAG Hallucination Detector")


GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

print("Loading embedding model...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)
client = Groq(api_key=GROQ_API_KEY)
print("Ready!")

class QuestionRequest(BaseModel):
    question: str

class RAGResponse(BaseModel):
    question: str
    answer: str
    score: float
    verdict: str
    reason: str
    unsupported_claims: str
    pages: list
    time_taken: float


def generate_answer(question):
    all_chunks = vectorstore.similarity_search(question, k=8)
    chunks = [c for c in all_chunks if c.metadata.get("page", 0) != 0][:4]
    context = "\n\n".join([c.page_content for c in chunks])

    prompt = f"""You are a precise assistant answering questions about AI agents.

RULES:
- Use ONLY information from the context below
- Be specific — reference actual details, examples, and terms from the context
- If the context has partial information, use what is there
- Do NOT invent facts, but DO explain what the context says fully

Context:
{context}

Question: {question}
Answer:"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content, context, chunks


def score_faithfulness(question, answer, context):
    prompt = f"""You are an expert at detecting hallucinations in AI answers.

Score the answer based on how well it is supported by the context.

Context:
{context}

Question: {question}
Answer: {answer}

Respond in exactly this format:
SCORE: [number between 0 and 1]
VERDICT: [Faithful / Partially Faithful / Hallucinated]
REASON: [one sentence]
UNSUPPORTED CLAIMS: [specific claims not in context, or "None"]"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content


def parse_field(text, field):
    for line in text.split("\n"):
        if line.startswith(f"{field}:"):
            return line.replace(f"{field}:", "").strip()
    return ""


def parse_score(text):
    for line in text.split("\n"):
        if line.startswith("SCORE:"):
            try:
                return float(line.replace("SCORE:", "").strip())
            except:
                pass
    return 0.5


# ── API Routes ────────────────────────────────────────────────

@app.get("/")
def serve_ui():
    return FileResponse("static/index.html")

@app.post("/ask", response_model=RAGResponse)
def ask(request: QuestionRequest):
    start = time.time()

    answer, context, chunks = generate_answer(request.question)
    scoring_output = score_faithfulness(request.question, answer, context)

    score = parse_score(scoring_output)
    verdict = parse_field(scoring_output, "VERDICT")
    reason = parse_field(scoring_output, "REASON")
    unsupported = parse_field(scoring_output, "UNSUPPORTED CLAIMS")
    pages = sorted(set([c.metadata.get("page", 0) for c in chunks]))

    return RAGResponse(
        question=request.question,
        answer=answer,
        score=score,
        verdict=verdict,
        reason=reason,
        unsupported_claims=unsupported,
        pages=pages,
        time_taken=round(time.time() - start, 1)
    )

app.mount("/static", StaticFiles(directory="static"), name="static")
