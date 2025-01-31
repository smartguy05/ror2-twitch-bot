import os
import openai
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Configure your Chroma client (using an in-memory or local DB)
client = chromadb.Client(
    Settings(
        persist_directory="./chroma_db",  # Directory to persist the database
        anonymized_telemetry=False       # Optional: Disable telemetry
    )
)


collection_name = "ror2_wiki"
collection = client.get_or_create_collection(
    name=collection_name,
    embedding_function=embedding_functions.OpenAIEmbeddingFunction(
        api_key=OPENAI_API_KEY,
        model_name="text-embedding-ada-002"
    )
)

def chunk_text(text, max_chars=1000):
    """
    Splits text into smaller chunks.
    """
    chunks = []
    current_chunk = []
    current_length = 0

    lines = text.split("\n")
    for line in lines:
        if current_length + len(line) > max_chars:
            chunks.append("\n".join(current_chunk))
            current_chunk = [line]
            current_length = len(line)
        else:
            current_chunk.append(line)
            current_length += len(line)
    if current_chunk:
        chunks.append("\n".join(current_chunk))

    return chunks

def build_index(input_file="wiki_data.txt"):
    print("Starting indexing")
    print("Reading file")
    # Read the entire file of scraped pages
    with open(input_file, "r", encoding="utf-8") as f:
        wiki_text = f.read()

    # Split by pages (as saved by wiki_scraper.py)
    pages = wiki_text.split("---PAGE:")
    doc_id_counter = 1
    chunk_num=0

    print("Processing pages")
    for page in pages:
        page = page.strip()
        if not page:
            continue

        # The first line might be something like " /wiki/...---..."
        # We can parse the page name from that if we want:
        lines = page.split("\n")
        page_name = lines[0]  # e.g. "/wiki/SomePage---"

        # Remainder is the text content
        text_content = "\n".join(lines[1:])
        chunks = chunk_text(text_content)

        # Add each chunk to the index
        for chunk in chunks:
            chunk_num=chunk_num+1
            print("Processing chunk " + str(chunk_num))
            doc_id = f"doc_{doc_id_counter}"
            collection.add(documents=[chunk], ids=[doc_id], metadatas=[{"page": page_name}])
            doc_id_counter += 1

    # Optionally, persist the DB if desired.
    print("Index built. Documents added:", doc_id_counter - 1)

if __name__ == "__main__":
    build_index()
