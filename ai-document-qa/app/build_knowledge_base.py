from pdf_loader import extract_text
from chunker import split_text
from vector_store import create_vector_store

path = "C:/Users/me_co/OneDrive/文档/AI-Engineering_Project/ai-document-qa/data/computer.pdf"

text = extract_text(path)

chunk = split_text(text)

vector = create_vector_store(chunk)

print(text[:1000])
print("Number of chunks:", len(chunk))
print("Vector data created successfully")