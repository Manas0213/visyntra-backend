import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

print("🧠 Visyntra's Brain Update Initiated...")

# 1. Saare PDFs ko 'therapy_data' folder se load karo
print("📂 Loading PDFs from 'therapy_data' folder...")
loader = PyPDFDirectoryLoader("therapy_data")
documents = loader.load()
print(f"✅ Loaded {len(documents)} pages in total.")

# 2. Data ko chhote chunks mein todo (taaki Llama-3 confuse na ho)
print("✂️ Chunking the data into smaller pieces...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, 
    chunk_overlap=200, # Thoda overlap rakha hai taaki context miss na ho
    length_function=len
)
chunks = text_splitter.split_documents(documents)
print(f"✅ Created {len(chunks)} knowledge chunks.")

# 3. Embeddings banao (Sentence ko numbers mein convert karo)
print("🧬 Generating Vector Embeddings (This might take a minute or two)...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 4. Naya FAISS Vector Database banao aur save karo
db = FAISS.from_documents(chunks, embeddings)

# Purana index overwrite ho jayega is naye aur powerful index se
db.save_local("cbt_faiss_index")
print("🎉 SUCCESS! Visyntra's brain (cbt_faiss_index) has been fully updated with all frameworks!")