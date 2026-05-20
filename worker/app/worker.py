import logging

from sqlmodel import Session

from src.application.dtos.queue_dtos import SummarizeEpubRequest
from src.application.use_cases.queue_use_cases import SummarizeEpubUseCase
from src.infrastructure.ai.ollama_agent import AIAgent
from src.infrastructure.database.db import engine
from src.infrastructure.queue.rabbitmq import RabbitMQQueue, wait_for_rabbitmq
from src.infrastructure.repositories.postgres_repository import PostgresBookRepository
from src.infrastructure.repositories.summary_job_repository import SummaryJobRepository

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def process_summary_job(payload: dict) -> None:
    job_id = payload.get("job_id")
    if not job_id:
        raise ValueError("Invalid message payload: missing job_id")

    with Session(engine) as session:

        use_case = SummarizeEpubUseCase(
            ai_agent=AIAgent(),
            repository=PostgresBookRepository(session),
            summary_job_repository=SummaryJobRepository(session),
        )

        try:
            response = use_case.execute(SummarizeEpubRequest(job_id=job_id))
            logger.info(
                "Completed summary job %s for book %s (%d chapters)",
                job_id,
                response.book_id,
                response.chapters_summarized,
            )
        except Exception as e:
            logger.exception("Summary job %s failed", job_id)
            raise


def main() -> None:
    logger.info("Starting summary worker")
    wait_for_rabbitmq()
    RabbitMQQueue().consume(process_summary_job)


if __name__ == "__main__":
    main()
