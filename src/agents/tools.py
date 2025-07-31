
import math
import re
import os

import numexpr
from langchain_chroma import Chroma
from langchain_core.tools import BaseTool, tool
from langchain_openai import AzureOpenAIEmbeddings
from core import settings
from sqlalchemy import create_engine, text
from langchain_community.vectorstores import PGVector


def calculator_func(expression: str) -> str:
    """Calculates a math expression using numexpr.

    Useful for when you need to answer questions about math using numexpr.
    This tool is only for math questions and nothing else. Only input
    math expressions.

    Args:
        expression (str): A valid numexpr formatted math expression.

    Returns:
        str: The result of the math expression.
    """

    try:
        local_dict = {"pi": math.pi, "e": math.e}
        output = str(
            numexpr.evaluate(
                expression.strip(),
                global_dict={},  # restrict access to globals
                local_dict=local_dict,  # add common mathematical functions
            )
        )
        return re.sub(r"^\[|\]$", "", output)
    except Exception as e:
        raise ValueError(
            f'calculator("{expression}") raised error: {e}.'
            " Please try again with a valid numerical expression"
        )


calculator: BaseTool = tool(calculator_func)
calculator.name = "Calculator"


# Format retrieved documents
# def format_contexts(docs):
#     return "\n\n".join(doc.page_content for doc in docs)
def format_contexts(docs):
    segments = []
    count = 1
    for d in docs:
        metadata = getattr(d, 'metadata', {}) or {}
        source = metadata.get('filename', 'unknown')
        url = metadata.get('file_path', 'unknown')
        # url = os.path.basename(source) if source else 'unknown'
        page = metadata.get('page_number', 'N/A')
        content = getattr(d, 'page_content', '[No content]')
        segments.append(f"""
Document: [{url}]({source})  
Page Number: `{page}`  
Page Content:
``` 
{content}

---
"""
        )

        count += 1
    return "\n\n".join(segments)



def load_chroma_db():
    # Create the embedding function for our project description database
    try:
        embeddings = AzureOpenAIEmbeddings(
                        api_key=settings.AZURE_OPENAI_API_KEY,
                        azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
                        model=settings.AZURE_OPENAI_EMBEDDER,
                        api_version=settings.AZURE_OPENAI_API_VERSION,
        )

    except Exception as e:
        raise RuntimeError(
            "Failed to initialize AzureOpenAIEmbeddings. Ensure the OpenAI API key is set."
        ) from e

    # Load the stored vector database
    chroma_db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    retriever = chroma_db.as_retriever(search_kwargs={"k": 5})
    return retriever


def load_postgre_db():
    # Create the embedding function for our project description database
    try:
        embeddings = AzureOpenAIEmbeddings(
                        api_key=settings.AZURE_OPENAI_API_KEY,
                        azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
                        model=settings.AZURE_OPENAI_EMBEDDER,
                        api_version=settings.AZURE_OPENAI_API_VERSION,
        )

    except Exception as e:
        raise RuntimeError(
            "Failed to initialize AzureOpenAIEmbeddings. Ensure the OpenAI API key is set."
        ) from e

    pg_url = settings.PGVECTOR_URL
    # Load the stored vector database
    postgre_db = PGVector(
        embedding_function=embeddings,
        connection_string=pg_url,
        collection_name="SEPS",)
    retriever = postgre_db.as_retriever(search_kwargs={"k": 5})
    return retriever

# def database_search_func(query: str) -> str:
#     """Searches chroma_db for information in the company's handbook."""
#     # Get the chroma retriever
#     retriever = load_chroma_db()

#     # Search the database for relevant documents
#     documents = retriever.invoke(query)

#     # Format the documents into a string
#     context_str = format_contexts(documents)

#     return context_str

def database_search_func(query: str) -> str:
    """Searches PGVector DB for information in the company's handbook."""

    retriever = load_postgre_db()
    documents = retriever.invoke(query)
    context_str = format_contexts(documents)
    return context_str



def get_full_doc_text_func(file_name: str) -> str:
    """Return the complete text (HTML if available) of a document stored in langchain_pg_embedding.

    Args:
        file_name (str): Exact filename stored in cmetadata->>'filename'.

    Returns:
        str: Concatenated text of all pages or an error string if nothing found.
    """
    try:
        engine = create_engine(settings.PGVECTOR_URL)
    except Exception as e:
        return f"Database connection error: {e}"

    query = text("""
        WITH collection_uuid AS (
            SELECT uuid
            FROM langchain_pg_collection
            WHERE name = 'SEPS'
        )
        SELECT cmetadata->>'page_number' AS page_number,
               document AS doc,
               cmetadata->>'text_as_html' AS html
        FROM langchain_pg_embedding
        WHERE collection_id = (SELECT uuid FROM collection_uuid)
          AND cmetadata->>'filename' = :file_name
        ORDER BY (cmetadata->>'page_number')::int;
    """)

    try:
        with engine.begin() as conn:
            rows = conn.execute(query, {"file_name": file_name}).mappings().all()
    except Exception as e:
        return f"Query execution failed: {e}"

    if not rows:
        return f"No document found for filename '{file_name}'."

    full_text_segments = []
    for row in rows:
        if row["html"]:
            full_text_segments.append(f"```html\n{row['html']}\n```")
        elif row["doc"]:
            full_text_segments.append(row["doc"])

    return "\n\n".join(full_text_segments)






database_search: BaseTool = tool(database_search_func)
database_search.name = "Database_Search"  # Update name with the purpose of your database


get_full_doc_text: BaseTool = tool(get_full_doc_text_func)
get_full_doc_text.name = "Get_Full_Doc_Text"
