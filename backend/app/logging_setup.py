from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        extra = getattr(record, "extra_fields", None)
        if extra:
            payload.update(extra)
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def configure_logging(level: int = logging.INFO) -> None:
    # Found running the local dev server on Windows: sys.stdout defaults to the
    # console's codepage (cp1252) even when redirected to a file, and
    # ensure_ascii=False deliberately writes raw UTF-8 (Devanagari transcripts,
    # etc.) rather than \uXXXX-escaping it. Every log line for a non-English query
    # crashed with UnicodeEncodeError - not fatal to the request (Python's logging
    # swallows handler errors), but it silently broke structured logging (spec
    # §12.7) for every language except English. Linux containers (Railway) default to
    # a UTF-8 locale already, so this was Windows-local-dev-specific, but reconfiguring
    # explicitly is the portable fix either way.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root.addHandler(handler)
