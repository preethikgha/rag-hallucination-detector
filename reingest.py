import fitz 
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import shutil, os


print("Extracting text with PyMuPDF...")
doc = fitz.open("docs/a-practical-guide-to-building-agents.pdf")
documents = []
for page_num, page in enumerate(doc):
    text = page.get_text()
    
    text = " ".join(text.split())  
    if len(text) > 100:  
        documents.append(Document(
            page_content=text,
            metadata={"page": page_num, "source": "agents_guide.pdf"}
        ))
    print(f"  Page {page_num}: {len(text)} chars")

print(f"\nExtracted {len(documents)} pages with real content")


splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(documents)
print(f"Created {len(chunks)} chunks")


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


results = vectorstore.similarity_search("when should you build an agent", k=3)
print("\nTest retrieval:")
for r in results:
    print(f"\n[Page {r.metadata['page']}] {r.page_content[:200]}")
