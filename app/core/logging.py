import json
import logging
from contextvars import ContextVar
from datetime import datetime, timezone


turn_id_context = ContextVar("talk_log_turn_id", default=None)


class JsonFormatter(logging.Formatter):
    def format(self, record):
        return json.dumps(
            {
                "time": datetime.now(timezone.utc).isoformat(),
                "level": record.levelname,
                "event": record.getMessage(),
                **getattr(record, "fields", {}),
            },
            ensure_ascii=False,
        )


def configure_logging():
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    logging.basicConfig(level=logging.INFO, handlers=[handler], force=True)
    logging.getLogger("httpx").setLevel(logging.WARNING)


def event(name: str, **fields):
    if (turn_id := turn_id_context.get()) is not None:
        fields.setdefault("turn_id", turn_id)
    logging.getLogger("ai-support").info(name, extra={"fields": fields})
