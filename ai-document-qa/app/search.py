from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

embedding_model=HuggingFaceEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2"
)

vector_store=Chroma(
    persist_directory="data/chroma_db",
    embedding_function=embedding_model

)

query="what is this document about?"

results=vector_store.similarity_search(query,k=3)

for i,result in enumerate(results):
    print(f"\n --- Result {i+1} --- ")
    print(result.page_content)