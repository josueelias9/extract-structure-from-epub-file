from typing import Any

from app.core.config import settings

from src.application.ports.queue_ports import QueuePort

# Lazy import avoids loading DB/AI dependencies in the extractor process.
from src.infrastructure.queue.tasks import process_summary_job_task


class CeleryQueue(QueuePort):
    """Celery adapter that sends summary jobs via RabbitMQ broker."""

    def publish(self, payload: dict[str, Any]) -> None:
        job_id = payload.get("job_id")
        if not job_id:
            raise ValueError("Invalid payload: missing job_id")

        process_summary_job_task.apply_async(
            kwargs={"job_id": job_id},
            queue=settings.SUMMARY_QUEUE_NAME,
        )
