from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

# ---------------------------------------------------------------------------
# Summarize
# ---------------------------------------------------------------------------


@dataclass
class EnqueueSummaryJobRequest:
    book_id: str
    chapter_ids: Optional[List[str]] = None  # None → all included chapters


@dataclass
class SummarizeEpubRequest:
    book_id: Optional[str] = None
    job_id: Optional[str] = None
    chapter_ids: Optional[List[str]] = None  # None → all included chapters


@dataclass
class SummarizeEpubResponse:
    book_id: str
    chapters_summarized: int


# ---------------------------------------------------------------------------
# LLM connectivity
# ---------------------------------------------------------------------------


@dataclass
class CheckLLMConnectionResponse:
    connected: bool
    host: str
    model: str
