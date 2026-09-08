import fitz
import os
from llama_index.core import Document, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

data_folder = "data"
documents = []

print("Extracting text from PDFs using PyMuPDF...")
for filename in os.listdir(data_folder):
    if filename.endswith(".pdf"):
        path = os.path.join(data_folder, filename)
        doc = fitz.open(path)
        full_text = ""
        for page in doc:
            full_text += page.get_text()
        documents.append(Document(text=full_text, metadata={"file_name": filename}))
        print(f"  Extracted {filename}: {len(full_text)} characters")

print(f"\nTotal documents: {len(documents)}")

print("Setting up embedding model (GPU)...")
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5", device="cuda")

print("Building the vector index...")
index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)

print("Saving index to storage/ ...")
index.storage_context.persist("storage")

print("Done. Your documents are now searchable with real extracted text.")