import logging
from datetime import datetime

from src.application.ports.service_ports import (
    BookRepositoryPort,
)

from src.application.ports.queue_ports import AIServicePort, SummaryJobRepositoryPort
from src.application.dtos.queue_dtos import (
    SummarizeEpubRequest,
    SummarizeEpubResponse,
    EnqueueSummaryJobRequest,
    CheckLLMConnectionResponse,
)

import logging

logger = logging.getLogger(__name__)


class EnqueueSummaryJobUseCase:
    def __init__(self, repository, job_repository, queue):
        self.repository = repository
        self.job_repository = job_repository
        self.queue = queue

    def execute(self, dto: EnqueueSummaryJobRequest):
        book_id = dto.book_id
        chapter_ids = dto.chapter_ids
        if self.repository.get_book(book_id) is None:
            raise Exception(f"Book {book_id!r} not found.")
        if chapter_ids:
            chapters_total = len(
                [
                    ch
                    for chapter_id in chapter_ids
                    if (ch := self.repository.get_chapter(chapter_id)) is not None
                    and ch.include
                ]
            )
        else:
            chapters_total = len(
                self.repository.get_chapters(book_id, include_only=True)
            )
        job = self.job_repository.create_job(
            book_id=book_id,
            chapter_ids=chapter_ids,
            chapters_total=chapters_total,
        )
        try:
            self.queue.publish({"job_id": job.id})
        except Exception as e:
            self.job_repository.mark_failed(job.id, f"Queue publish failed: {e}")
            raise Exception("Failed to enqueue summarization job.")
        return {
            "job_id": job.id,
            "book_id": job.book_id,
            "status": job.status,
            "chapters_total": job.chapters_total,
            "chapters_summarized": job.chapters_summarized,
            "error_message": job.error_message,
        }


class GetSummaryJobStatusUseCase:
    def __init__(self, job_repository):
        self.job_repository = job_repository

    def execute(self, job_id):
        job = self.job_repository.get_job(job_id)
        if job is None:
            raise Exception(f"Job {job_id!r} not found.")
        return {
            "job_id": job.id,
            "book_id": job.book_id,
            "status": job.status,
            "chapters_total": job.chapters_total,
            "chapters_summarized": job.chapters_summarized,
            "error_message": job.error_message,
        }


class GetLatestSummaryJobUseCase:
    def __init__(self, job_repository):
        self.job_repository = job_repository

    def execute(self, book_id):
        job = self.job_repository.get_latest_for_book(book_id)
        if job is None:
            raise Exception("No summary job found for this book.")
        return {
            "job_id": job.id,
            "book_id": job.book_id,
            "status": job.status,
            "chapters_total": job.chapters_total,
            "chapters_summarized": job.chapters_summarized,
            "error_message": job.error_message,
        }


# ---------------------------------------------------------------------------
# Summarize
# ---------------------------------------------------------------------------


class SummarizeEpubUseCase:
    """Generate AI summaries for included chapters and persist them.
    If a job_id is provided, the summarisation will be tracked in the SummaryJobRepository.
    """

    def __init__(
        self,
        ai_agent: AIServicePort,
        repository: BookRepositoryPort,
        summary_job_repository: SummaryJobRepositoryPort,
    ):
        self._ai_agent = ai_agent
        self._repository = repository
        self._summary_job_repository = summary_job_repository

    def execute(self, request: SummarizeEpubRequest) -> SummarizeEpubResponse:
        job_id = request.job_id
        book_id = request.book_id
        chapter_ids = request.chapter_ids

        if job_id:
            book_id, chapter_ids = self._summary_job_repository.begin_job(job_id)

        if not book_id:
            raise ValueError("book_id is required when job_id is not provided")

        logger.info("Starting summarisation for book %r", book_id)

        try:
            if chapter_ids:
                # Summarise a specific subset — still honour the include flag
                chapters = [
                    ch
                    for cid in chapter_ids
                    if (ch := self._repository.get_chapter(cid)) is not None
                    and ch.include
                ]
            else:
                # Summarise all chapters that are flagged for inclusion
                chapters = self._repository.get_chapters(book_id, include_only=True)

            count = 0
            for chapter in chapters:
                if not chapter.content:
                    logger.info("Skipping chapter %r (no content)", chapter.id)
                    continue
                logger.info("Summarising chapter %r — %s", chapter.id, chapter.title)
                try:
                    summary = self._ai_agent.summarize_content(chapter.content)
                except Exception as e:
                    logger.error("Failed to summarise chapter %r: %s", chapter.id, e)
                    continue
                self._repository.update_chapter_summary(
                    chapter_id=chapter.id,
                    summary=summary,
                    summary_date=datetime.utcnow(),
                    ai_generated=True,
                )
                count += 1

            if job_id:
                self._summary_job_repository.mark_completed(job_id, count)

            logger.info("Summarisation complete — %d chapters processed", count)
            return SummarizeEpubResponse(book_id=book_id, chapters_summarized=count)
        except Exception as e:
            if job_id:
                self._summary_job_repository.mark_failed(job_id, str(e))
            raise


# ---------------------------------------------------------------------------
# Check LLM connection
# ---------------------------------------------------------------------------


class CheckLLMConnectionUseCase:
    """Verify that the Ollama LLM service is reachable."""

    def __init__(self, ai_agent: AIServicePort):
        self._ai_agent = ai_agent

    def execute(self) -> CheckLLMConnectionResponse:
        info = self._ai_agent.get_connection_info()
        logger.info(
            "Checking LLM connection to %s (model: %s)", info["host"], info["model"]
        )
        ok = self._ai_agent.test_connection()
        return CheckLLMConnectionResponse(
            connected=ok, host=info["host"], model=info["model"]
        )
