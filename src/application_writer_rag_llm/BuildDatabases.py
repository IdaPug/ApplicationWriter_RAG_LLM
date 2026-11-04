
import hashlib
import json
import os
import re
import shutil
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    BSHTMLLoader,
)


# ============================================================
# 1. PROJECT PATHS AND CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Load environment variables from the project root.
load_dotenv(PROJECT_ROOT / ".env")


APPLICATIONS_DIR = PROJECT_ROOT / "data" / "Applications"

GUIDELINES_DIR = PROJECT_ROOT / "data" / "WritingGuidelines"
LECTURE_DIR = GUIDELINES_DIR / "Lecture_guidelines"

CHROMA_DIR = PROJECT_ROOT / "data" / "chromaDB"

APPLICATIONS_DB_DIR = CHROMA_DIR / "applications_db"
GUIDELINES_DB_DIR = CHROMA_DIR / "guidelines_db"
JOB_POSTINGS_DB_DIR = CHROMA_DIR / "job_postings_db"

# Store generated summaries separately from the vector databases.
JOB_SUMMARIES_DIR = PROJECT_ROOT / "data" / "job_posting_summaries"


# ------------------------------------------------------------
# Embedding model
# ------------------------------------------------------------

# Use the same embedding model when querying all three databases.
EMBEDDING_MODEL = (
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)


# ------------------------------------------------------------
# Text splitting
# ------------------------------------------------------------

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


# ------------------------------------------------------------
# Supported file types
# ------------------------------------------------------------

SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".txt",
    ".md",
    ".html",
    ".htm",
}


# ------------------------------------------------------------
# Job posting identification
# ------------------------------------------------------------

# File names are normalized before comparison.
# For example:
# JobPosting.pdf       -> jobposting
# job_posting.txt      -> jobposting
# Job Posting.html     -> jobposting

JOB_POSTING_NAMES = {
    "jobposting",
    "jobdescription",
    "jobad",
    "jobadvertisement",
    "jobannouncement",
    "joboffer",
    "jobopening",
    "jobopportunity",
    "positiondescription",
    "vacancy",
    "jobopslag",
    "stillingsopslag",
    "jobannonce",
}


# ------------------------------------------------------------
# Summary configuration
# ------------------------------------------------------------

# Set this to True to regenerate every summary.
# Normally, summaries are regenerated only when the source
# job posting has changed.
FORCE_REGENERATE_SUMMARIES = False

# Keep this prompt stable to avoid unnecessary changes.
SUMMARY_PROMPT_VERSION = "1"

SUMMARY_MAX_WORDS = 400


# ============================================================
# 2. INITIALISE EMBEDDINGS AND TEXT SPLITTER
# ============================================================

print("\nInitialising embedding model...")

embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""],
)


# ============================================================
# 3. HELPER FUNCTIONS
# ============================================================

def numeric_folder_sort(path: Path):
    """
    Sort folders numerically:
    App_1, App_2, App_10 rather than App_1, App_10, App_2.
    """

    match = re.search(r"(\d+)$", path.name)

    if match:
        return int(match.group(1))

    return float("inf")


def normalize_filename(filename: str) -> str:
    """
    Remove spaces, underscores, hyphens and punctuation
    to make file-name matching more robust.
    """

    return re.sub(
        r"[^a-z0-9]",
        "",
        Path(filename).stem.lower(),
    )


def is_job_posting(file_path: Path) -> bool:
    """
    Determine whether a file is a job posting based on its name.

    Add additional names to JOB_POSTING_NAMES if your files
    use different naming conventions.
    """

    normalized_name = normalize_filename(file_path.name)

    return normalized_name in JOB_POSTING_NAMES


def load_document(file_path: Path):
    """
    Load a document based on its file extension.

    Returns a list of LangChain Document objects.
    """

    extension = file_path.suffix.lower()

    if extension == ".pdf":
        loader = PyPDFLoader(str(file_path))

    elif extension in {".txt", ".md"}:
        loader = TextLoader(
            str(file_path),
            encoding="utf-8",
            autodetect_encoding=True,
        )

    elif extension in {".html", ".htm"}:
        loader = BSHTMLLoader(
            str(file_path),
            open_encoding="utf-8",
        )

    else:
        return []

    documents = loader.load()

    for document in documents:
        document.metadata["source"] = str(
            file_path.relative_to(PROJECT_ROOT)
        )
        document.metadata["file_name"] = file_path.name

    return documents


def load_application_metadata(application_dir: Path):
    """
    Load metadata.json for an application.
    Returns an empty dictionary if it does not exist.
    """

    metadata_path = application_dir / "metadata.json"

    if not metadata_path.exists():
        print(
            f"  Warning: No metadata.json in {application_dir.name}"
        )
        return {}

    try:
        with open(metadata_path, "r", encoding="utf-8") as file:
            return json.load(file)

    except (json.JSONDecodeError, OSError) as error:
        print(f"  Warning: Could not read metadata: {error}")
        return {}


def add_metadata(documents, metadata):
    """
    Add application-level metadata to every document.
    """

    for document in documents:
        for key, value in metadata.items():

            # Chroma metadata values should be simple scalar values.
            if isinstance(value, (str, int, float, bool)):
                document.metadata[key] = value

    return documents


def split_documents(documents):
    """
    Split documents into smaller chunks.
    """

    return text_splitter.split_documents(documents)


def rebuild_database(documents, database_path, collection_name):
    """
    Delete the old database and create a fresh collection.

    This prevents old documents from remaining after files
    are changed or removed.
    """

    if database_path.exists():
        shutil.rmtree(database_path)

    database_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not documents:
        print(f"No documents found for {collection_name}.")
        return

    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=str(database_path),
    )

    print(f"Created collection: {collection_name}")
    print(f"Stored documents/chunks: {len(documents)}")
    print(f"Database location: {database_path}")


# ============================================================
# 4. OPENROUTER CLIENT
# ============================================================

def initialize_llm():
    """
    Initialise the OpenRouter client.

    Requires the following variables in the project .env:

        OPENROUTER_API_KEY=...
        OPENROUTER_MODEL=...
    """

    api_key = os.getenv("OPENROUTER_API_KEY")
    model = os.getenv("OPENROUTER_MODEL")

    if not api_key:
        raise ValueError(
            "OPENROUTER_API_KEY was not found in the .env file."
        )

    if not model:
        raise ValueError(
            "OPENROUTER_MODEL was not found in the .env file."
        )

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        default_headers={
            "HTTP-Referer": "http://localhost",
            "X-OpenRouter-Title": "Application Writer RAG LLM",
        },
    )

    return client, model


# ============================================================
# 5. JOB POSTING SUMMARY FUNCTIONS
# ============================================================

def calculate_text_hash(text: str) -> str:
    """
    Calculate a SHA-256 hash of the original job posting.

    This is used to determine whether the posting has changed.
    """

    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


def get_summary_paths(
    application_id: str,
    file_name: str,
):
    """
    Return the summary and metadata paths for a job posting.

    The file name is included so that an application can
    contain more than one job posting.
    """

    safe_file_name = re.sub(
        r"[^a-zA-Z0-9_-]",
        "_",
        Path(file_name).stem,
    )

    summary_name = (
        f"{application_id}__{safe_file_name}"
    )

    summary_path = JOB_SUMMARIES_DIR / f"{summary_name}.md"

    metadata_path = JOB_SUMMARIES_DIR / f"{summary_name}.json"

    return summary_path, metadata_path


def load_cached_summary(
    summary_path: Path,
    metadata_path: Path,
    source_hash: str,
):
    """
    Return a cached summary if the original posting has not
    changed.

    Otherwise, return None.
    """

    if FORCE_REGENERATE_SUMMARIES:
        return None

    if not summary_path.exists() or not metadata_path.exists():
        return None

    try:
        with open(metadata_path, "r", encoding="utf-8") as file:
            cached_metadata = json.load(file)

        cached_hash = cached_metadata.get("source_hash")

        cached_prompt_version = cached_metadata.get(
            "prompt_version"
        )

        if (
            cached_hash == source_hash
            and cached_prompt_version == SUMMARY_PROMPT_VERSION
        ):
            return summary_path.read_text(encoding="utf-8")

    except (json.JSONDecodeError, OSError):
        return None

    return None


def generate_job_posting_summary(
    llm: OpenAI,
    model: str,
    job_posting_text: str,
):
    """
    Generate a structured summary of a job posting using
    the configured OpenRouter model.

    The summary is designed for semantic similarity search.
    """

    system_prompt = """
You are an expert at analysing job postings.

Your task is to create a concise, structured summary of a
job posting. This summary will later be used to identify
previous job applications that are relevant to a new job.

Focus on the actual nature of the position and the skills,
responsibilities, and qualifications that matter when
comparing job opportunities.

Important rules:

1. Only include information supported by the job posting.
2. Never invent requirements, responsibilities, or skills.
3. Preserve important technical terminology.
4. Include both explicit requirements and desirable skills.
5. Distinguish required skills from preferred skills.
6. Ignore generic application instructions and boilerplate.
7. Preserve the original technical meaning.
8. Write the summary in English, even if the original
   job posting is in Danish.
9. Keep important Danish or English technical terms where
   they improve searchability.
10. Do not evaluate the applicant or suggest qualifications.

Use the following structure:

# Job Posting Summary

## Job Title and Organisation

## Role Description

## Main Responsibilities

## Required Technical Skills

## Preferred Technical Skills

## Domain Knowledge

## Desired Candidate Profile

## Important Keywords

Omit sections for which the job posting provides no
relevant information.

Keep the summary concise and informative.
Aim for approximately 200-400 words.
"""

    user_prompt = f"""
Summarise the following job posting.

Do not add information that is not present in the source.

JOB POSTING:

{job_posting_text}
"""

    response = llm.chat.completions.create(
        model=model,
        temperature=0.1,
        max_tokens=1000,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
    )

    summary = response.choices[0].message.content

    if not summary or not summary.strip():
        raise ValueError(
            "The LLM returned an empty job posting summary."
        )

    return summary.strip()


def save_summary(
    summary: str,
    summary_path: Path,
    metadata_path: Path,
    source_hash: str,
):
    """
    Save a generated summary and its metadata.

    The metadata stores the source hash so that unchanged
    job postings do not need to be summarised again.
    """

    JOB_SUMMARIES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_path.write_text(
        summary,
        encoding="utf-8",
    )

    metadata = {
        "source_hash": source_hash,
        "prompt_version": SUMMARY_PROMPT_VERSION,
    }

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=4,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def create_job_posting_summary_document(
    application_id: str,
    application_metadata: dict,
    file_path: Path,
    job_posting_documents: list,
    llm: OpenAI,
    model: str,
):
    """
    Create one summary Document for one job posting.

    All pages of the original posting are combined before
    summarisation.

    The resulting Document contains the summary and is
    intended to produce exactly one embedding.
    """

    # Combine all pages of the original job posting.
    job_posting_text = "\n\n".join(
        document.page_content
        for document in job_posting_documents
        if document.page_content.strip()
    )

    if not job_posting_text.strip():
        print(
            f"  Warning: Empty job posting: {file_path.name}"
        )
        return None

    source_hash = calculate_text_hash(job_posting_text)

    summary_path, metadata_path = get_summary_paths(
        application_id,
        file_path.name,
    )

    # First check whether a valid summary already exists.
    summary = load_cached_summary(
        summary_path,
        metadata_path,
        source_hash,
    )

    if summary is not None:
        print(
            f"  Using cached summary: {file_path.name}"
        )

    else:
        print(
            f"  Generating summary: {file_path.name}"
        )

        summary = generate_job_posting_summary(
            llm=llm,
            model=model,
            job_posting_text=job_posting_text,
        )

        save_summary(
            summary=summary,
            summary_path=summary_path,
            metadata_path=metadata_path,
            source_hash=source_hash,
        )

        print(
            f"  Saved summary: {summary_path.name}"
        )

    # Create one LangChain Document for the entire posting.
    metadata = {
        "application_id": application_id,
        "document_type": "job_posting_summary",
        "original_document_type": file_path.stem,
        "file_name": file_path.name,
        "source": str(file_path.relative_to(PROJECT_ROOT)),
        "summary_path": str(
            summary_path.relative_to(PROJECT_ROOT)
        ),
    }

    # Add application-level metadata.
    for key, value in application_metadata.items():
        if isinstance(value, (str, int, float, bool)):
            metadata[key] = value

    return Document(
        page_content=summary,
        metadata=metadata,
    )


# ============================================================
# 6. BUILD APPLICATIONS DATABASE
# ============================================================

def build_applications_database():

    print("\n" + "=" * 60)
    print("BUILDING APPLICATIONS DATABASE")
    print("=" * 60)

    all_documents = []

    application_folders = sorted(
        [
            folder
            for folder in APPLICATIONS_DIR.iterdir()
            if folder.is_dir()
        ],
        key=numeric_folder_sort,
    )

    print(
        f"Found {len(application_folders)} application folders."
    )

    for application_dir in application_folders:

        print(f"\nProcessing {application_dir.name}...")

        application_metadata = load_application_metadata(
            application_dir
        )

        application_documents = []

        for file_path in sorted(application_dir.rglob("*")):

            if not file_path.is_file():
                continue

            if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue

            try:
                documents = load_document(file_path)

                documents = add_metadata(
                    documents,
                    application_metadata,
                )

                for document in documents:

                    document.metadata["application_id"] = (
                        application_dir.name
                    )

                    if is_job_posting(file_path):
                        document.metadata["document_type"] = (
                            "job_posting"
                        )
                    else:
                        document.metadata["document_type"] = (
                            file_path.stem
                        )

                application_documents.extend(documents)

                print(f"  Loaded: {file_path.name}")

            except Exception as error:
                print(
                    f"  Could not load {file_path.name}: {error}"
                )

        all_documents.extend(application_documents)

    # Split all application documents into chunks.
    chunks = split_documents(all_documents)

    print(
        f"\nTotal application documents: {len(all_documents)}"
    )
    print(
        f"Total application chunks: {len(chunks)}"
    )

    rebuild_database(
        documents=chunks,
        database_path=APPLICATIONS_DB_DIR,
        collection_name="applications",
    )


# ============================================================
# 7. BUILD JOB POSTINGS DATABASE
# ============================================================

def build_job_postings_database():

    print("\n" + "=" * 60)
    print("BUILDING JOB POSTINGS DATABASE")
    print("=" * 60)

    if not APPLICATIONS_DIR.exists():
        print(
            f"Applications directory not found: {APPLICATIONS_DIR}"
        )
        return

    application_folders = sorted(
        [
            folder
            for folder in APPLICATIONS_DIR.iterdir()
            if folder.is_dir()
        ],
        key=numeric_folder_sort,
    )

    # First identify all job postings.
    posting_candidates = []

    for application_dir in application_folders:

        for file_path in sorted(application_dir.rglob("*")):

            if not file_path.is_file():
                continue

            if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue

            if is_job_posting(file_path):
                posting_candidates.append(
                    (application_dir, file_path)
                )

    print(
        f"Found {len(posting_candidates)} job posting files."
    )

    if not posting_candidates:
        print(
            "No job postings found. Check JOB_POSTING_NAMES "
            "and your file names."
        )

        rebuild_database(
            documents=[],
            database_path=JOB_POSTINGS_DB_DIR,
            collection_name="job_postings",
        )
        return

    # Initialise OpenRouter only if summaries need to be generated.
    llm = None
    model = None

    summary_documents = []

    for application_dir, file_path in posting_candidates:

        application_id = application_dir.name

        print(
            f"\nProcessing {application_id}: {file_path.name}"
        )

        application_metadata = load_application_metadata(
            application_dir
        )

        try:
            job_posting_documents = load_document(file_path)

            if not job_posting_documents:
                print(
                    f"  Warning: No text extracted from {file_path.name}"
                )
                continue

            # Check whether a cached summary exists before
            # initialising the LLM.
            original_text = "\n\n".join(
                document.page_content
                for document in job_posting_documents
                if document.page_content.strip()
            )

            source_hash = calculate_text_hash(original_text)

            summary_path, metadata_path = get_summary_paths(
                application_id,
                file_path.name,
            )

            cached_summary = load_cached_summary(
                summary_path,
                metadata_path,
                source_hash,
            )

            if cached_summary is None:

                if llm is None:
                    print(
                        "\nInitialising OpenRouter for summary generation..."
                    )
                    llm, model = initialize_llm()

                summary = generate_job_posting_summary(
                    llm=llm,
                    model=model,
                    job_posting_text=original_text,
                )

                save_summary(
                    summary=summary,
                    summary_path=summary_path,
                    metadata_path=metadata_path,
                    source_hash=source_hash,
                )

                print(
                    f"  Generated and saved: {summary_path.name}"
                )

            else:
                summary = cached_summary

                print(
                    f"  Reusing cached summary: {summary_path.name}"
                )

            # One summary document = one embedding.
            summary_document = Document(
                page_content=summary,
                metadata={
                    "application_id": application_id,
                    "document_type": "job_posting_summary",
                    "original_document_type": file_path.stem,
                    "file_name": file_path.name,
                    "source": str(
                        file_path.relative_to(PROJECT_ROOT)
                    ),
                    "summary_path": str(
                        summary_path.relative_to(PROJECT_ROOT)
                    ),
                    **{
                        key: value
                        for key, value in application_metadata.items()
                        if isinstance(
                            value,
                            (str, int, float, bool),
                        )
                    },
                },
            )

            summary_documents.append(summary_document)

        except Exception as error:
            print(
                f"  Could not process {file_path.name}: {error}"
            )

    print(
        f"\nTotal job posting summaries: {len(summary_documents)}"
    )

    rebuild_database(
        documents=summary_documents,
        database_path=JOB_POSTINGS_DB_DIR,
        collection_name="job_postings",
    )


# ============================================================
# 8. BUILD GUIDELINES DATABASE
# ============================================================

def build_guidelines_database():

    print("\n" + "=" * 60)
    print("BUILDING GUIDELINES DATABASE")
    print("=" * 60)

    all_documents = []

    if not LECTURE_DIR.exists():
        print(
            f"Lecture directory not found: {LECTURE_DIR}"
        )
        return

    lecture_files = sorted(
        LECTURE_DIR.rglob("*")
    )

    for file_path in lecture_files:

        if not file_path.is_file():
            continue

        if file_path.suffix.lower() not in {
            ".md",
            ".txt",
            ".pdf",
        }:
            continue

        try:
            documents = load_document(file_path)

            for document in documents:
                document.metadata["source_type"] = "lecture"
                document.metadata["document_type"] = "lecture"
                document.metadata["topic"] = file_path.stem

            all_documents.extend(documents)

            print(f"  Loaded: {file_path.name}")

        except Exception as error:
            print(
                f"  Could not load {file_path.name}: {error}"
            )

    chunks = split_documents(all_documents)

    print(
        f"\nTotal guideline documents: {len(all_documents)}"
    )
    print(
        f"Total guideline chunks: {len(chunks)}"
    )

    rebuild_database(
        documents=chunks,
        database_path=GUIDELINES_DB_DIR,
        collection_name="guidelines",
    )


# ============================================================
# 9. MAIN
# ============================================================

def main():

    print("\nAPPLICATION WRITER - DATABASE BUILDER")
    print("=" * 60)

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Embedding model: {EMBEDDING_MODEL}")

    # Build the chunked application database.
    build_applications_database()

    # Build the document-level job posting summary database.
    build_job_postings_database()

    # Build the chunked guidelines database.
    build_guidelines_database()

    print("\n" + "=" * 60)
    print("DATABASE BUILD COMPLETED")
    print("=" * 60)

    print("\nDatabase locations:")
    print(f"  Applications: {APPLICATIONS_DB_DIR}")
    print(f"  Job postings: {JOB_POSTINGS_DB_DIR}")
    print(f"  Guidelines:  {GUIDELINES_DB_DIR}")

    print("\nSummary files:")
    print(f"  {JOB_SUMMARIES_DIR}")


if __name__ == "__main__":
    main()
