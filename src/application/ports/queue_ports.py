from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from src.enterprise.entities import Book, Chapter

# ---------------------------------------------------------------------------
# AI service
# ---------------------------------------------------------------------------


class AIServicePort(ABC):
    """Port for AI text summarisation and connectivity checks."""

    @abstractmethod
    def summarize_content(self, content: str) -> str: ...

    @abstractmethod
    def test_connection(self) -> bool: ...

    @abstractmethod
    def get_connection_info(self) -> Dict[str, str]:
        """Return {'host': ..., 'model': ...} for diagnostics."""
        ...


class SummaryJobRepositoryPort(ABC):
    """Persistence port for async summarisation jobs."""

    @abstractmethod
    def begin_job(self, job_id: str) -> tuple[str, Optional[List[str]]]:
        """Mark a job as processing and return (book_id, chapter_ids)."""
        ...

    @abstractmethod
    def mark_completed(self, job_id: str, chapters_summarized: int): ...

    @abstractmethod
    def mark_failed(self, job_id: str, error_message: str): ...


# ---------------------------------------------------------------------------
# Queue
# ---------------------------------------------------------------------------


class QueuePort(ABC):
    """Port for sending summary jobs to an async task queue."""

    @abstractmethod
    def publish(self, payload: Dict[str, str]) -> None:
        """Send a job payload to the queue. Raises on failure."""
        ...
