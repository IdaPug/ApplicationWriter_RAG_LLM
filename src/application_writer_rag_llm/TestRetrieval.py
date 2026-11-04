
"""
Test retrieval from the ApplicationWriter_RAG_LLM Chroma databases.

Run from the project root:
    uv run src/application_writer_rag_llm/TestRetrieval.py
"""

from pathlib import Path

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


# ============================================================
# 1. CONFIGURATION
# ============================================================

# Expected location:
# project_root/src/application_writer_rag_llm/TestRetrieval.py

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

DATABASES_DIR = PROJECT_ROOT / "data" / "chromaDB"

APPLICATIONS_DB_DIR = DATABASES_DIR / "applications_db"
GUIDELINES_DB_DIR = DATABASES_DIR / "guidelines_db"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

APPLICATIONS_COLLECTION = "applications"
GUIDELINES_COLLECTION = "guidelines"

DEFAULT_K = 5


# ============================================================
# 2. LOAD EMBEDDINGS
# ============================================================

def create_embeddings():
    """Load the same embedding model used during database creation."""

    print("\nLoading embedding model...")

    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


# ============================================================
# 3. LOAD DATABASES
# ============================================================

def load_database(database_dir, collection_name, embeddings):
    """Open an existing persistent Chroma database."""

    if not database_dir.exists():
        raise FileNotFoundError(
            f"Database not found: {database_dir}\n"
            "Run BuildDatabases.py first."
        )

    return Chroma(
        collection_name=collection_name,
        persist_directory=str(database_dir),
        embedding_function=embeddings,
    )


# ============================================================
# 4. DISPLAY DATABASE INFORMATION
# ============================================================

def show_database_counts(applications_db, guidelines_db):

    print("\n" + "=" * 70)
    print("DATABASE INFORMATION")
    print("=" * 70)

    for name, database in [
        ("Applications", applications_db),
        ("Guidelines", guidelines_db),
    ]:
        count = database._collection.count()
        print(f"{name}: {count} chunks")


# ============================================================
# 5. RETRIEVAL
# ============================================================

def search_database(database, query, database_name, k=5):

    print("\n" + "=" * 70)
    print(f"SEARCHING: {database_name}")
    print("=" * 70)

    print(f"Query: {query}\n")

    results = database.similarity_search_with_score(
        query,
        k=k,
    )

    if not results:
        print("No results found.")
        return

    for i, (document, distance) in enumerate(results, start=1):

        metadata = document.metadata or {}

        print("-" * 70)
        print(f"RESULT {i}")
        print("-" * 70)

        print(f"Distance: {distance:.4f}")
        print(f"Source: {metadata.get('source', 'Unknown')}")
        print(
            f"Document type: "
            f"{metadata.get('document_type', 'Unknown')}"
        )

        if metadata.get("application_id"):
            print(f"Application ID: {metadata['application_id']}")

        if metadata.get("company"):
            print(f"Company: {metadata['company']}")

        if metadata.get("job_type"):
            print(f"Job type: {metadata['job_type']}")

        if metadata.get("language"):
            print(f"Language: {metadata['language']}")

        if metadata.get("outcome"):
            print(f"Outcome: {metadata['outcome']}")

        if metadata.get("self_ranking") is not None:
            print(f"Self-ranking: {metadata['self_ranking']}")

        if metadata.get("page") is not None:
            print(f"Page: {metadata['page']}")

        print("\nCHUNK TEXT:\n")

        # Limit displayed text to keep the terminal readable.
        text = document.page_content

        print(text[:1800])

        if len(text) > 1800:
            print("\n[Text truncated]")

        print()


# ============================================================
# 6. MAIN
# ============================================================

def main():

    print("=" * 70)
    print("APPLICATION WRITER - RETRIEVAL TEST")
    print("=" * 70)

    print(f"\nProject root: {PROJECT_ROOT}")

    embeddings = create_embeddings()

    print("\nOpening databases...")

    applications_db = load_database(
        APPLICATIONS_DB_DIR,
        APPLICATIONS_COLLECTION,
        embeddings,
    )

    guidelines_db = load_database(
        GUIDELINES_DB_DIR,
        GUIDELINES_COLLECTION,
        embeddings,
    )

    show_database_counts(
        applications_db,
        guidelines_db,
    )

    print("\nExample queries:")
    print("1. Machine learning and computer vision experience")
    print("2. Previous applications for data scientist positions")
    print("3. How to write a convincing cover letter")
    print("4. Examples of demonstrating technical skills")

    while True:

        print("\n" + "-" * 70)

        query = input(
            "\nEnter search query (or 'q' to quit): "
        ).strip()

        if query.lower() in {"q", "quit", "exit"}:
            break

        if not query:
            print("Please enter a search query.")
            continue

        k_input = input(
            f"Number of results per database [{DEFAULT_K}]: "
        ).strip()

        try:
            k = int(k_input) if k_input else DEFAULT_K

            if k < 1:
                raise ValueError

        except ValueError:
            print("Invalid number. Using default value.")
            k = DEFAULT_K

        # Search previous applications
        search_database(
            applications_db,
            query,
            "APPLICATIONS DATABASE",
            k,
        )

        # Search lecture and guidelines
        search_database(
            guidelines_db,
            query,
            "GUIDELINES DATABASE",
            k,
        )

    print("\nRetrieval test finished.")


if __name__ == "__main__":
    main()
