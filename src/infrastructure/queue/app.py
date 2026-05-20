import os

from celery import Celery

from app.core.config import settings

app = Celery("summary_worker", broker=settings.CELERY_BROKER_URL)

app.conf.update(
    task_default_queue=settings.SUMMARY_QUEUE_NAME,
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    enable_utc=True,
    timezone="UTC",
    task_ignore_result=True,
    include=["src.infrastructure.queue.tasks"],
)

if os.getenv("CELERY_TASK_ALWAYS_EAGER", "false").lower() == "true":
    app.conf.task_always_eager = True
