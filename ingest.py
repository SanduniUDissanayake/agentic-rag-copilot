from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

print("Loading documents from data/ ...")
documents = SimpleDirectoryReader("data").load_data()
print(f"Loaded {len(documents)} document chunks from source files.")

print("Setting up local embedding model (first run downloads it, may take a few minutes)...")
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5", device="cuda")

print("Building the vector index (this is the slow part — embedding every chunk)...")
index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)

print("Saving index to storage/ ...")
index.storage_context.persist("storage")

print("Done. Your documents are now searchable.")