# ApplicationWriter RAG LLM

A Retrieval-Augmented Generation (RAG) application for creating tailored
job applications, CV profiles, and skills sections using previous
applications, personal background information, and writing guidelines.

The project uses Chroma vector databases for document retrieval and
an LLM accessed through OpenRouter for text generation.

## What the project does

Given a new job posting, the application:

1. Loads the job posting from the configured input folder.
2. Retrieves similar historical job postings from `job_postings_db`.
3. Uses the retrieved application IDs to find corresponding cover letters
   in `applications_db`.
4. Retrieves relevant writing and lecture guidance from `guidelines_db`.
5. Combines the retrieved context with the personal profile and writing rules.
6. Sends the prompt to the configured OpenRouter model.
7. Saves the generated document(s).

The application can generate:
- A tailored job application / cover letter.
- A tailored CV profile.
- A tailored skills section.
- All three outputs.

## data used:
- Applications: One folder for each previously application containing CV, Coverletter and jobposting (.txt or .html)
- WritingGuidelines: Guidelines for how to write an application
    - lecture_guidelines: files containing guides and examples from a lecture on how to write an application
    - writing_rules.md: Rules given to the LLM to follow when generating the application
    - personal_profile.md: Personal profile of the applicant. Could be text from linkedin, Course completed, achivements or other skils


## Project structure

```text
ApplicationWriter_RAG_LLM/
├── config/
│   └── config.yaml
├── data/
│   ├── Applications/
│   │   ├── App_1/
│   │   ├── App_2/
│   ├── WritingGuidelines/
│   │   ├── lecture_guidelines/
│   │   ├── writing_rules.md
│   │   ├── personal_profile.md
│   ├── chromaDB/
│   │   ├── applications_db/
│   │   ├── guidelines_db/
│   │   └── job_postings_db/
│   ├── job_posting_summaries/
│   ├── input/
│   └── output/
├── src/
│   └── application_writer_rag_llm/
│       ├── DataProcessing.py 
│       ├── BuildDatabases.py
│       └── application_writer.py
├── .env
├── .gitignore
└── pyproject.toml