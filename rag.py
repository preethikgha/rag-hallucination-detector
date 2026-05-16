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

def ask(question):
    
    all_chunks = vectorstore.similarity_search(question, k=8)
    chunks = [c for c in all_chunks if c.metadata.get("page", 0) != 0][:4]
    context = "\n\n".join([c.page_content for c in chunks])

   
    prompt = f"""You are a helpful assistant answering questions about AI agents.
Use the context below to answer. Be specific and detailed using the information given.
If something is partially covered, explain what you do know from the context.

Context:
{context}

Question: {question}

Answer:"""


    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )

    answer = response.choices[0].message.content

    print(f"\nQuestion: {question}")
    print(f"\nAnswer: {answer}")
    print(f"\n--- Retrieved from pages: {[c.metadata.get('page') for c in chunks]} ---")


ask("What is an agent?")
ask("When should you build an agent?")
ask("What are guardrails in agents?")
