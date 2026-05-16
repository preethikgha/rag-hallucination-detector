import fitz  # pymupdf
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import shutil, os

# Step 1: Extract text using PyMuPDF (much better than PyPDF)
print("Extracting text with PyMuPDF...")
doc = fitz.open("docs/a-practical-guide-to-building-agents.pdf")
documents = []
for page_num, page in enumerate(doc):
    text = page.get_text()
    # Clean up the garbled spacing
    text = " ".join(text.split())  # collapses weird whitespace
    if len(text) > 100:  # skip nearly empty pages
        documents.append(Document(
            page_content=text,
            metadata={"page": page_num, "source": "agents_guide.pdf"}
        ))
    print(f"  Page {page_num}: {len(text)} chars")

print(f"\nExtracted {len(documents)} pages with real content")

# Step 2: Chunk
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(documents)
print(f"Created {len(chunks)} chunks")

# Step 3: Re-embed and store
print("Re-embedding...")
if os.path.exists("./chroma_db"):
    shutil.rmtree("./chroma_db")

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)
print("Done! Database rebuilt with clean text.")

# Step 4: Quick test
results = vectorstore.similarity_search("when should you build an agent", k=3)
print("\nTest retrieval:")
for r in results:
    print(f"\n[Page {r.metadata['page']}] {r.page_content[:200]}")