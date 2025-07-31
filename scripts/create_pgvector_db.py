
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import PGVector
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
import truststore

truststore.inject_into_ssl()
load_dotenv()

# Pull your full connection string from env
PG_URL = os.environ["PGVECTOR_URL"]

# (Optional) if you want to rebuild from scratch each time:
# PRE_DELETE = True
PRE_DELETE = True

# 1) Set up your embeddings
embeddings = AzureOpenAIEmbeddings(
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    model=os.environ["AZURE_OPENAI_EMBEDDER"],
    api_version=os.environ["AZURE_OPENAI_API_VERSION"],
)

def create_pgvector_db(folder_path: str, collection_name: str = "pgvector_embeddings"):
    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=500)

    for fname in os.listdir(folder_path):
        path = os.path.join(folder_path, fname)
        if not (fname.endswith(".pdf") or fname.endswith(".docx")):
            continue

        loader = PyPDFLoader(path) if fname.endswith(".pdf") else Docx2txtLoader(path)
        docs = loader.load()
        chunks = splitter.split_documents(docs)

        # This will create the extension, tables, and collection on first run
        PGVector.from_documents(
            documents=chunks,
            embedding=embeddings,
            connection_string=PG_URL,
            collection_name=collection_name,
            pre_delete_collection=PRE_DELETE,  # drop+recreate if you need a fresh table
            use_jsonb=True,                   # recommended for metadata
        )
        print(f"Ingested {len(chunks)} chunks from {fname}")

    # Return a live PGVector instance for querying
    return PGVector(
        connection_string=PG_URL,
        embedding_function=embeddings,
        collection_name=collection_name,
        use_jsonb=True,
    )

if __name__ == "__main__":
    retriever = create_pgvector_db("./data")
    # Wrap in a LangChain retriever if you like:
    retriever = retriever.as_retriever(search_kwargs={"k": 3})

    # Test it:
    results = retriever.get_relevant_documents("mission and values")
    for doc in results:
        print("—", doc.page_content[:200].replace("\n", " "), "…")

