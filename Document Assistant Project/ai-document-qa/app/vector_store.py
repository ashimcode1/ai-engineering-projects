from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


def create_vector_store(chunks):

    embedding_model = HuggingFaceEmbeddings(
        model="sentence-transformers/all-MiniLM-L6-v2"
    )

    metadatas = [
        {
            "source": "computer_basics.pdf",
            "chunk_id": i
        }
        for i in range(len(chunks))
    ]

    ids = [
        f"computer_basics.pdf_chunk_{i}"
        for i in range(len(chunks))
    ]

    vector_store = Chroma(
        collection_name="computer_basics",
        embedding_function=embedding_model,
        persist_directory="data/chroma_db"
    )

    vector_store.add_texts(
        texts=chunks,
        metadatas=metadatas,
        ids=ids
    )

    return vector_store


