from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Any

import yaml
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"

# Fallback path for job_postings_db if it is not configured in config.yaml.
from application_writer_rag_llm.BuildDatabases import JOB_POSTINGS_DB_DIR

load_dotenv(PROJECT_ROOT / ".env")


def load_config() -> dict[str, Any]:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Configuration file not found: {CONFIG_PATH}")
    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError("Configuration file must contain a YAML mapping.")
    return config


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


# ---------------------------------------------------------------------
# Input job posting
# ---------------------------------------------------------------------

def read_text_file(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return path.read_text(encoding="utf-8")


def read_html_file(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    soup = BeautifulSoup(path.read_text(encoding="utf-8"), "html.parser")
    for element in soup(["script", "style", "noscript", "header", "footer", "nav"]):
        element.decompose()
    return soup.get_text(separator="\\n", strip=True)


def load_job_posting(config: dict[str, Any]) -> tuple[str, Path]:
    input_config = config["input"]
    input_directory = resolve_path(input_config["directory"])
    if not input_directory.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_directory}")

    extensions = {ext.lower() for ext in input_config["supported_extensions"]}
    files = sorted(
        f for f in input_directory.iterdir()
        if f.is_file() and f.suffix.lower() in extensions
    )

    if not files:
        raise FileNotFoundError(f"No supported job posting found in: {input_directory}")
    if len(files) > 1:
        raise ValueError(
            "More than one job posting was found. Keep only one supported file.\\n"
            + "\\n".join(f.name for f in files)
        )

    job_file = files[0]
    job_text = (
        read_html_file(job_file)
        if job_file.suffix.lower() in {".html", ".htm"}
        else read_text_file(job_file)
    )
    if not job_text.strip():
        raise ValueError(f"The job posting is empty: {job_file}")

    print(f"Loaded job posting: {job_file.name}")
    return job_text, job_file


# ---------------------------------------------------------------------
# Embeddings and Chroma
# ---------------------------------------------------------------------

def load_embeddings(config: dict[str, Any]) -> HuggingFaceEmbeddings:
    embedding_config = config["embeddings"]
    return HuggingFaceEmbeddings(
        model_name=embedding_config["model_name"],
        model_kwargs={"device": embedding_config.get("device", "cpu")},
        encode_kwargs={"normalize_embeddings": True},
    )


def load_vector_database(
    database_config: dict[str, Any],
    embeddings: HuggingFaceEmbeddings,
) -> Chroma:
    directory = resolve_path(database_config["directory"])
    if not directory.exists():
        raise FileNotFoundError(f"Chroma database not found: {directory}")
    return Chroma(
        collection_name=database_config["collection_name"],
        persist_directory=str(directory),
        embedding_function=embeddings,
    )


def get_job_postings_database_config(config: dict[str, Any]) -> dict[str, Any]:
    configured = config.get("databases", {}).get("jobpostings")
    if configured:
        return configured
    return {
        "directory": str(JOB_POSTINGS_DB_DIR),
        "collection_name": "job_postings",
    }


# ---------------------------------------------------------------------
# Fixed writing context
# ---------------------------------------------------------------------

def load_writing_guidance(config: dict[str, Any]) -> tuple[str, str]:
    guidance = config["guidelines"]
    writing_rules = read_text_file(resolve_path(guidance["writing_rules"]))
    personal_profile = read_text_file(resolve_path(guidance["personal_profile"]))
    return writing_rules, personal_profile


# ---------------------------------------------------------------------
# Retrieval: similar job-posting summaries -> application IDs
# ---------------------------------------------------------------------

def retrieve_similar_applications(
    job_postings_db: Chroma,
    job_text: str,
    config: dict[str, Any],
) -> list[Document]:
    """Retrieve summary documents from the dedicated job_postings collection."""
    k = config["retrieval"].get("similar_applications", 3)
    results = job_postings_db.similarity_search(query=job_text, k=k)
    return [
        doc for doc in results
        if doc.metadata.get("application_id") is not None
    ]


def get_unique_application_ids(documents: list[Document]) -> list[str]:
    """Extract unique IDs in similarity-ranking order."""
    ids: list[str] = []
    seen: set[str] = set()
    for doc in documents:
        value = doc.metadata.get("application_id")
        if value is None:
            continue
        application_id = str(value)
        if application_id not in seen:
            seen.add(application_id)
            ids.append(application_id)
    return ids


# ---------------------------------------------------------------------
# Retrieve previous cover letters from applications_db
# ---------------------------------------------------------------------

def retrieve_previous_application_documents(
    applications_db: Chroma,
    selected_application_ids: list[str],
    config: dict[str, Any],
) -> list[Document]:
    if not selected_application_ids:
        return []

    k = config["retrieval"]["previous_application_chunks"]
    retrieved: list[Document] = []

    for application_id in selected_application_ids:
        try:
            docs = applications_db.similarity_search(
                query="motivation relevant experience qualifications skills",
                k=k,
                filter={
                    "$and": [
                        {"application_id": application_id},
                        {"document_type": "cover_letter"},
                    ]
                },
            )
            retrieved.extend(docs)
        except Exception as error:
            print(f"Warning: retrieval failed for {application_id}: {error}")

    return retrieved


def retrieve_guideline_chunks(
    guidelines_db: Chroma,
    job_text: str,
    config: dict[str, Any],
) -> list[Document]:
    return guidelines_db.similarity_search(
        query=job_text,
        k=config["retrieval"]["guideline_chunks"],
    )


# ---------------------------------------------------------------------
# Prompt context formatting
# ---------------------------------------------------------------------

def format_documents(
    documents: list[Document],
    max_characters: int = 18000,
) -> str:
    if not documents:
        return "No relevant documents were retrieved."

    sections = []
    for index, doc in enumerate(documents, start=1):
        application_id = doc.metadata.get("application_id", "Unknown")
        source = doc.metadata.get("source", "Unknown source")
        sections.append(
            f"DOCUMENT {index}\\n"
            f"Application ID: {application_id}\\n"
            f"Source: {source}\\n\\n"
            f"{doc.page_content}"
        )
    return "\\n\\n".join(sections)[:max_characters]


# ---------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------

def build_system_prompt(
    config: dict[str, Any],
    writing_rules: str,
    personal_profile: str,
) -> str:
    generation = config["generation"]
    return f"""
You are an expert job application and CV writing assistant.

Create professional, tailored career documents based on the user's real
background and the supplied job posting.

OUTPUT TYPE:
{generation["output_type"]}

OUTPUT LANGUAGE:
{generation["language"]}

WRITING RULES:
{writing_rules}

PERSONAL PROFILE:
{personal_profile}

Use only facts supported by the personal profile and supplied documents.
Do not invent experience, qualifications, skills, achievements, or results.
If information is missing, do not fabricate it.
"""


def build_user_prompt(
    job_text: str,
    previous_applications: str,
    lecture_guidance: str,
    config: dict[str, Any],
) -> str:
    generation = config["generation"]
    output_type = generation["output_type"]
    language = generation["language"]
    application = generation["application"]
    cv_profile = generation["cv_profile"]
    skills = generation["skills"]

    common = f"""
JOB POSTING:
{job_text}

RELEVANT PREVIOUS APPLICATIONS:
{previous_applications}

RETRIEVED LECTURE GUIDANCE:
{lecture_guidance}
"""

    if output_type == "application":
        sections = []
        if application["include_motivation"]:
            sections.append("Explain why the role and organization are motivating.")
        if application["include_relevant_experience"]:
            sections.append("Present relevant experience and qualifications.")
        if application["include_skills"]:
            sections.append("Highlight relevant skills supported by the profile.")
        if application["include_closing"]:
            sections.append("Finish with a professional and appropriate closing.")
        instructions = "\\n".join(f"- {section}" for section in sections)
        return f"""
{common}

TASK:
Write a tailored job application / cover letter in {language}.

Follow these instructions:
{instructions}

Maximum length: {application["max_words"]} words.
Write the complete application as a polished document.
Do not include commentary about your process.
"""

    if output_type == "cv_profile":
        voice = (
            "Write in the first person."
            if cv_profile["use_first_person"]
            else "Do not use first-person pronouns. Use a concise professional profile style."
        )
        return f"""
{common}

TASK:
Write a tailored CV profile text in {language}.
Maximum length: {cv_profile["max_words"]} words.
{voice}

The profile should:
- Present the candidate's most relevant professional identity.
- Emphasize qualifications that match the job posting.
- Be specific and evidence-based.
- Avoid generic claims and unsupported statements.
- Be suitable for the top section of a CV.

Return only the CV profile text.
"""

    if output_type == "skills":
        explanation = (
            "For each skill, briefly explain how it is supported by the candidate's background or why it is relevant."
            if skills["include_explanations"]
            else "Return a concise list of skills without explanations."
        )
        return f"""
{common}

TASK:
Create a tailored skills section in {language}.
Select at most {skills["max_skills"]} skills.
Prioritize skills explicitly requested in the job posting and supported by
the personal profile. Do not invent skills or claim proficiency without evidence.
{explanation}
Return a clean, CV-ready skills section.
"""

    if output_type == "all":
        return f"""
{common}

TASK:
Generate all three outputs in {language}:

# APPLICATION
A complete tailored job application / cover letter, maximum
{application["max_words"]} words.

# CV PROFILE
A tailored CV profile, maximum {cv_profile["max_words"]} words.
Use first person: {cv_profile["use_first_person"]}.

# SKILLS
A tailored skills section, maximum {skills["max_skills"]} skills.
Include explanations: {skills["include_explanations"]}.

Use the candidate's real background and do not invent qualifications.
Use the exact headings shown above so the outputs can be split into files.
"""

    raise ValueError(
        f"Unsupported output_type: {output_type}. "
        "Choose application, cv_profile, skills, or all."
    )


# ---------------------------------------------------------------------
# OpenRouter client and generation
# ---------------------------------------------------------------------

def initialize_llm(config: dict[str, Any]) -> OpenAI:
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        raise FileNotFoundError(f".env file not found: {env_path}")

    load_dotenv(env_path, override=True)
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    model_name = os.getenv("OPENROUTER_MODEL", "").strip()

    if not api_key or api_key == "placeholder":
        raise ValueError(
            "OPENROUTER_API_KEY is missing or still set to 'placeholder' in .env."
        )
    if not model_name:
        raise ValueError("OPENROUTER_MODEL is missing from .env.")

    print(f"Using OpenRouter model: {model_name}")
    return OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        default_headers={
            "HTTP-Referer": "http://localhost",
            "X-OpenRouter-Title": "ApplicationWriter_RAG_LLM",
        },
    )


def generate_document(
    llm: OpenAI,
    system_prompt: str,
    user_prompt: str,
    config: dict[str, Any],
) -> str:
    response = llm.chat.completions.create(
        model=os.getenv("OPENROUTER_MODEL"),
        temperature=config["llm"].get("temperature", 0.3),
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response.choices[0].message.content or ""


# ---------------------------------------------------------------------
# Output parsing and saving
# ---------------------------------------------------------------------

def split_all_output(generated_text: str) -> dict[str, str]:
    patterns = {
        "application": r"(?im)^#+\\s*APPLICATION\\s*$",
        "cv_profile": r"(?im)^#+\\s*CV PROFILE\\s*$",
        "skills": r"(?im)^#+\\s*SKILLS\\s*$",
    }

    positions = {}
    for name, pattern in patterns.items():
        match = re.search(pattern, generated_text)
        if match:
            positions[name] = match.start()

    if not positions:
        return {"application": generated_text}

    ordered = sorted(positions.items(), key=lambda item: item[1])
    result = {}
    for index, (name, start) in enumerate(ordered):
        end = ordered[index + 1][1] if index + 1 < len(ordered) else len(generated_text)
        section = generated_text[start:end]
        result[name] = re.sub(patterns[name], "", section, count=1).strip()
    return result


def save_output(
    generated_text: str,
    config: dict[str, Any],
    job_file: Path,
) -> list[Path]:
    generation = config["generation"]
    output_type = generation["output_type"]
    output_directory = resolve_path(generation["output_directory"])
    output_directory.mkdir(parents=True, exist_ok=True)

    prefix = generation["filename_prefix"]
    safe_job_name = re.sub(r"[^a-zA-Z0-9_-]+", "_", job_file.stem).strip("_")
    safe_job_name = safe_job_name or "job_posting"
    base = f"{prefix}_{safe_job_name}"

    if output_type == "all":
        sections = split_all_output(generated_text)
        paths = []
        for section_type in ("application", "cv_profile", "skills"):
            content = sections.get(section_type, "").strip()
            if not content:
                continue
            path = output_directory / f"{base}_{section_type}.md"
            path.write_text(content, encoding="utf-8")
            paths.append(path)
        return paths

    path = output_directory / f"{base}_{output_type}.md"
    path.write_text(generated_text, encoding="utf-8")
    return [path]


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    print("\\nApplication Writer\\n==================\\n")
    config = load_config()
    generation = config["generation"]
    output_type = generation["output_type"]

    print(f"Output type: {output_type}")
    print(f"Language: {generation['language']}")

    job_text, job_file = load_job_posting(config)
    writing_rules, personal_profile = load_writing_guidance(config)

    print("\\nLoading embedding model...")
    embeddings = load_embeddings(config)

    print("Loading Chroma databases...")
    applications_db = load_vector_database(
        config["databases"]["applications"], embeddings
    )
    guidelines_db = load_vector_database(
        config["databases"]["guidelines"], embeddings
    )
    job_postings_db = load_vector_database(
        get_job_postings_database_config(config), embeddings
    )

    # The dedicated summary database now determines which old applications
    # are relevant. No additional LLM-based selection step is necessary.
    print("\\nRetrieving similar previous job postings...")
    similar_job_documents = retrieve_similar_applications(
        job_postings_db, job_text, config
    )
    selected_application_ids = get_unique_application_ids(similar_job_documents)

    print(f"Retrieved {len(similar_job_documents)} job-posting summaries.")
    print("Relevant previous applications:")
    if selected_application_ids:
        for application_id in selected_application_ids:
            print(f"  - {application_id}")
    else:
        print("  No matching previous applications found.")

    print("\\nRetrieving previous application documents...")
    previous_documents = retrieve_previous_application_documents(
        applications_db, selected_application_ids, config
    )
    previous_context = format_documents(previous_documents)

    print("\\nRetrieving lecture guidance...")
    guideline_documents = retrieve_guideline_chunks(
        guidelines_db, job_text, config
    )
    guideline_context = format_documents(
        guideline_documents, max_characters=12000
    )

    system_prompt = build_system_prompt(
        config, writing_rules, personal_profile
    )
    user_prompt = build_user_prompt(
        job_text, previous_context, guideline_context, config
    )

    llm = initialize_llm(config)

    print(f"\\nGenerating {output_type}...")
    generated_text = generate_document(
        llm, system_prompt, user_prompt, config
    )
    if not generated_text.strip():
        raise RuntimeError("The language model returned an empty response.")

    output_paths = save_output(generated_text, config, job_file)
    print("\\nGeneration completed.")
    for path in output_paths:
        print(f"Saved: {path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print("\\nERROR\\n-----")
        print(error)
        sys.exit(1)
