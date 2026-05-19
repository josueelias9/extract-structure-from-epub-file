#!/usr/bin/env python3
"""Smoke test for the summary worker.

What this script does:
1) Ensures DB tables exist.
2) Seeds a sample book + chapters if missing.
3) Creates a summary job.
4) Executes the real worker handler `process_summary_job`.
5) Verifies job status and saved chapter summaries.

By default it patches the AI agent with a deterministic fake implementation so
you can validate the worker flow without depending on Ollama.
"""

from __future__ import annotations

import argparse
import os
import sys
import uuid
from typing import Iterable


def _configure_env_defaults() -> None:
    """Provide sane local defaults when environment variables are missing."""
    defaults = {
        "DB_HOST": "localhost",
        "DB_PORT": "5432",
        "POSTGRES_USER": "postgres",
        "POSTGRES_PASSWORD": "postgres",
        "POSTGRES_DB": "postgres",
        "LLM_HOST": "http://localhost:11434",
        "LLM_MODEL": "llama3.1",
    }
    for key, value in defaults.items():
        os.environ.setdefault(key, value)


_configure_env_defaults()

from sqlmodel import Session

from app import worker as worker_module
from src.enterprise.entities import Book, Chapter
from src.infrastructure.database.db import engine, init_db
from src.infrastructure.repositories.postgres_repository import PostgresBookRepository
from src.infrastructure.repositories.summary_job_repository import SummaryJobRepository


class FakeAIAgent:
    """Deterministic replacement for the real Ollama-based agent."""

    def summarize_content(self, content: str) -> str:
        words = content.split()
        if len(words) <= 24:
            return "[fake-summary] " + content.strip()
        return "[fake-summary] " + " ".join(words[:24]) + " ..."


def _ensure_seed_book(session: Session, book_id: str) -> tuple[str, int]:
    repo = PostgresBookRepository(session)

    if repo.get_book(book_id) is None:
        repo.save_book(
            Book(
                id=book_id,
                name="Worker Smoke Test Book",
                language="es",
                author="smoke-test",
            )
        )

    existing = repo.get_chapters(book_id)
    if not existing:
        chapter_1_id = f"ch-{uuid.uuid4()}"
        chapter_2_id = f"ch-{uuid.uuid4()}"
        repo.save_chapters(
            [
                Chapter(
                    id=chapter_1_id,
                    book_id=book_id,
                    title="Capitulo 1",
                    number="1",
                    include=True,
                    content=(
                        "Este es un contenido de prueba para validar el worker. "
                        "Tiene suficiente longitud para que pase por el flujo de resumen "
                        "y escriba un resultado en base de datos sin usar Ollama real."
                    ),
                ),
                Chapter(
                    id=chapter_2_id,
                    book_id=book_id,
                    title="Capitulo 2 (excluido)",
                    number="2",
                    include=False,
                    content="Este capitulo no deberia resumirse porque include=False.",
                ),
            ]
        )

    included = repo.get_chapters(book_id, include_only=True)
    return book_id, len(included)


def _create_job(session: Session, book_id: str, chapters_total: int) -> str:
    jobs = SummaryJobRepository(session)
    job = jobs.create_job(
        book_id=book_id,
        chapter_ids=None,
        chapters_total=chapters_total,
    )
    return job.id


def _count_summarized_chapters(session: Session, book_id: str) -> int:
    repo = PostgresBookRepository(session)
    chapters = repo.get_chapters(book_id, include_only=True)
    return sum(1 for ch in chapters if ch.summary)


def run_smoke_test(book_id: str, use_real_ai: bool) -> None:
    init_db()

    if not use_real_ai:
        worker_module.AIAgent = FakeAIAgent

    with Session(engine) as session:
        book_id, included_count = _ensure_seed_book(session, book_id)
        if included_count == 0:
            raise RuntimeError("No included chapters found for smoke test")
        job_id = _create_job(session, book_id, included_count)

    worker_module.process_summary_job({"job_id": job_id})

    with Session(engine) as session:
        jobs = SummaryJobRepository(session)
        job = jobs.get_job(job_id)
        if job is None:
            raise RuntimeError(f"Job {job_id} was not found after processing")
        if job.status != "completed":
            raise RuntimeError(
                f"Job {job_id} ended with status={job.status!r} error={job.error_message!r}"
            )

        summarized = _count_summarized_chapters(session, book_id)
        if summarized <= 0:
            raise RuntimeError("No summaries were saved for included chapters")

        print("SMOKE TEST OK")
        print(f"book_id={book_id}")
        print(f"job_id={job_id}")
        print(f"job_status={job.status}")
        print(f"chapters_summarized={job.chapters_summarized}")
        print(f"included_chapters_with_summary={summarized}")


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run worker smoke test")
    parser.add_argument(
        "--book-id",
        default="worker-smoke-book",
        help="Book id used for smoke test data",
    )
    parser.add_argument(
        "--use-real-ai",
        action="store_true",
        help="Use real Ollama AIAgent instead of deterministic fake agent",
    )
    return parser.parse_args(list(argv))


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    try:
        run_smoke_test(book_id=args.book_id, use_real_ai=args.use_real_ai)
        return 0
    except Exception as exc:
        print("SMOKE TEST FAILED")
        print(str(exc))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
