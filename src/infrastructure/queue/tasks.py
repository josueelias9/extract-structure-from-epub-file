import logging

from sqlmodel import Session

from src.application.dtos.queue_dtos import SummarizeEpubRequest
from src.application.use_cases.queue_use_cases import SummarizeEpubUseCase
from src.infrastructure.ai.ollama_agent import AIAgent
from src.infrastructure.database.db import engine
from src.infrastructure.queue.app import app
from src.infrastructure.repositories.postgres_repository import PostgresBookRepository
from src.infrastructure.repositories.summary_job_repository import SummaryJobRepository

logger = logging.getLogger(__name__)


@app.task(name="summary.process_job")
def process_summary_job_task(job_id: str) -> dict[str, int | str]:
    if not job_id:
        raise ValueError("Invalid payload: missing job_id")

    with Session(engine) as session:
        use_case = SummarizeEpubUseCase(
            ai_agent=AIAgent(),
            repository=PostgresBookRepository(session),
            summary_job_repository=SummaryJobRepository(session),
        )

        response = use_case.execute(SummarizeEpubRequest(job_id=job_id))
        logger.info(
            "Completed summary job %s for book %s (%d chapters)",
            job_id,
            response.book_id,
            response.chapters_summarized,
        )
        return {
            "job_id": job_id,
            "book_id": response.book_id,
            "chapters_summarized": response.chapters_summarized,
        }
