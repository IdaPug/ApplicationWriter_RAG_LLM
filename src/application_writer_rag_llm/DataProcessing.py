
from pathlib import Path
import json


# Set projectroot to the root of the project, assuming this script is in src/application_writer_rag_llm/
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

APPLICATIONS_DIR = Path(PROJECT_ROOT) / "data" / "applications"


# File extensions to recognize
CV_EXTENSIONS = {".pdf", ".docx", ".doc", ".txt"}
POSTING_EXTENSIONS = {".pdf", ".txt", ".html", ".htm", ".md"}
COVER_LETTER_EXTENSIONS = {".pdf", ".docx", ".doc", ".txt"}


def find_files(folder, keywords, extensions):
    """Find files whose names contain one of the given keywords."""
    matches = []

    for file in folder.iterdir():
        if not file.is_file():
            continue

        if file.suffix.lower() not in extensions:
            continue

        filename = file.stem.lower()

        if any(keyword in filename for keyword in keywords):
            matches.append(file)

    return matches


def get_file_paths(folder):
    """Identify CV, job posting, and cover letter files."""
    cv_files = find_files(
        folder,
        ["cv", "resume", "curriculum"],
        CV_EXTENSIONS,
    )

    posting_files = find_files(
        folder,
        ["job", "posting", "position", "vacancy", "description"],
        POSTING_EXTENSIONS,
    )

    cover_letter_files = find_files(
        folder,
        ["cover", "letter", "motivation", "application"],
        COVER_LETTER_EXTENSIONS,
    )

    # Return paths relative to the project root
    return {
        "cv": [str(path.relative_to(PROJECT_ROOT)) for path in cv_files],
        "job_posting": [
            str(path.relative_to(PROJECT_ROOT)) for path in posting_files
        ],
        "cover_letter": [
            str(path.relative_to(PROJECT_ROOT))
            for path in cover_letter_files
        ],
    }


def ask_value(prompt, default=""):
    """Ask for a value, allowing the user to leave it blank."""
    value = input(f"{prompt} [{default}]: ").strip()
    return value if value else default


def ask_self_ranking():
    """Ask for a ranking from 1 to 5, or allow it to be blank."""
    while True:
        value = input("Self-ranking (1-5, or Enter to skip): ").strip()

        if not value:
            return None

        if value in {"1", "2", "3", "4", "5"}:
            return int(value)

        print("Please enter a number from 1 to 5, or press Enter to skip.")


def create_metadata(folder):
    metadata_path = folder / "metadata.json"

    # Avoid overwriting existing metadata
    if metadata_path.exists():
        print(f"Skipping {folder.name}: metadata.json already exists.")
        return

    files = get_file_paths(folder)

    print("\n" + "=" * 50)
    print(f"Application: {folder.name}")

    print("\nDetected files:")
    print(f"  CVs: {files['cv'] or 'None detected'}")
    print(f"  Job postings: {files['job_posting'] or 'None detected'}")
    print(f"  Cover letters: {files['cover_letter'] or 'None detected'}")

    if not files["job_posting"]:
        print("\nNo job posting detected.")
        print("This may be an unsolicited application.")

    company = ask_value("Company")
    job_type = ask_value("Job type")
    language = ask_value("Language")

    outcome = ask_value(
    "Outcome (pending/interview/second_round/rejected/offer/no_response/other)",
    "pending",
    
    )

    self_ranking = ask_self_ranking()

    metadata = {
        "application_id": folder.name,
        "company": company,
        "job_type": job_type,
        "language": language,
        "outcome": outcome,
        "self_ranking": self_ranking,
        "files": files,
    }

    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4, ensure_ascii=False)

    print(f"\nCreated: {metadata_path}")


def main():
    if not APPLICATIONS_DIR.exists():
        print(f"Applications folder not found: {APPLICATIONS_DIR}")
        return

    application_folders = sorted(
    (
        folder
        for folder in APPLICATIONS_DIR.iterdir()
        if folder.is_dir()
        and folder.name.startswith("App_")
        and folder.name.removeprefix("App_").isdigit()
    ),
    key=lambda folder: int(folder.name.removeprefix("App_")),
    )

    if not application_folders:
        print("No App_ folders found.")
        return

    print(f"Found {len(application_folders)} application folders.")

    for folder in application_folders:
        create_metadata(folder)

    print("\nFinished creating metadata files.")

if __name__ == "__main__":
    main()
