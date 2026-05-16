from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from groq import Groq
import os
from dotenv import load_dotenv
load_dotenv()


GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)
client = Groq(api_key=GROQ_API_KEY)

# ============================================================
# STEP 1 — Generate answer (same as Phase 2)
# ============================================================

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
- Never say "the document does not cover this" if the context has related information

Context:
{context}

Question: {question}
Answer:"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )
    answer = response.choices[0].message.content
    return answer, context, chunks

# ============================================================
# STEP 2 — Score faithfulness (hallucination detection)
# ============================================================

def score_faithfulness(question, answer, context):
    scoring_prompt = f"""You are an expert at detecting hallucinations in AI-generated answers.

Your job: check if the ANSWER is fully supported by the CONTEXT below.

Score the answer on this scale:
- 1.0 = Perfectly faithful, every claim is in the context
- 0.7 = Mostly faithful, minor extrapolation
- 0.5 = Partially faithful, some claims not in context
- 0.3 = Mostly hallucinated, few claims supported
- 0.0 = Completely hallucinated, nothing from context

Context:
{context}

Question: {question}

Answer to evaluate:
{answer}

Respond in exactly this format:
SCORE: [number between 0 and 1]
VERDICT: [Faithful / Partially Faithful / Hallucinated]
REASON: [one sentence explaining your score]
UNSUPPORTED CLAIMS: [list any specific claims not found in context, or "None"]"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": scoring_prompt}]
    )
    return response.choices[0].message.content

# ============================================================
# STEP 3 — Parse the score and print result
# ============================================================

def parse_score(scoring_output):
    score = 0.5  # default
    for line in scoring_output.split("\n"):
        if line.startswith("SCORE:"):
            try:
                score = float(line.replace("SCORE:", "").strip())
            except:
                pass
    return score

def get_badge(score):
    if score >= 0.8:
        return "FAITHFUL"
    elif score >= 0.4:
        return "PARTIALLY FAITHFUL"
    else:
        return "HALLUCINATED"

# ============================================================
# STEP 4 — Full pipeline
# ============================================================

def ask_with_scoring(question):
    print("\n" + "="*60)
    print(f"Question: {question}")
    print("="*60)

    # Generate answer
    answer, context, chunks = generate_answer(question)
    print(f"\nAnswer:\n{answer}")
    print(f"\nRetrieved from pages: {[c.metadata.get('page') for c in chunks]}")

    # Score it
    print("\nScoring faithfulness...")
    scoring_output = score_faithfulness(question, answer, context)
    score = parse_score(scoring_output)
    badge = get_badge(score)

    print(f"\n{badge} (Score: {score})")
    print(f"\nDetailed scoring:\n{scoring_output}")

# ============================================================
# Run on 3 questions
# ============================================================

ask_with_scoring("What is an agent?")
ask_with_scoring("When should you build an agent?")
ask_with_scoring("What are guardrails in agents?")