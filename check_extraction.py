import fitz  # this is PyMuPDF's import name
import os

data_folder = "data"

for filename in os.listdir(data_folder):
    if filename.endswith(".pdf"):
        path = os.path.join(data_folder, filename)
        try:
            doc = fitz.open(path)
            full_text = ""
            for page in doc:
                full_text += page.get_text()
            print(f"File: {filename}")
            print(f"Pages: {len(doc)}")
            print(f"Text length: {len(full_text)} characters")
            print(f"First 300 characters: {full_text[:300]}")
            print("---")
        except Exception as e:
            print(f"File: {filename} — FAILED TO OPEN: {e}")
            print("---")