from langchain_ollama import ChatOllama
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

embedding_model=HuggingFaceEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2"
)

llm=ChatOllama(
    model="qwen2.5:0.5b",temperature=0
)

vector_store = Chroma(
    collection_name="computer_basics",
    persist_directory="data/chroma_db",
    embedding_function=embedding_model
)

print(vector_store._collection.configuration_json)

print("\n===== COLLECTION INFO =====")

print("Count:", vector_store._collection.count())

data = vector_store._collection.get(
    include=["documents", "metadatas"]
)

print("\n===== ALL STORED DOCUMENTS =====")

for i, (doc, metadata) in enumerate(
    zip(data["documents"], data["metadatas"])
):
    print(f"\n--- Stored Chunk {i} ---")
    print("Metadata:", metadata)
    print("Content:")
    print(doc[:1000])

question = input("Ask a question ")

print("\nDEBUG QUESTION:", repr(question))

results = vector_store.similarity_search_with_score(
    question,
    k=5
)

for i, (result, score) in enumerate(results):

    print(f"\n--- Result {i + 1} ---")

    print("Score:", score)

    print("Content:")
    print(result.page_content)

    print("Metadata:")
    print(result.metadata)

print("\nDEBUG: Retrieval finished")    

context="\n\n".join(result.page_content for result,score in results)

print("\nDEBUG: Context created")

prompt = f"""
Answer the question using only the context provided below.

If the answer cannot be found in the context, say:
"I don't know based on the document."

Context:
{context}

Question:
{question}

Answer: """

print("\n========== PROMPT ==========")
print(prompt)
print("============================")

response=llm.invoke(prompt)

print("\n========== RESPONSE ==========")
print(response.content)
print("==============================")