import logging
import json
import sys

from datetime import datetime, timezone
from pathlib import Path
from typing import Any


LOG_PATH = Path(__file__).resolve().parents[1] / "logs" / "api.log"
JSON_LOG_PATH = Path(__file__).resolve().parents[1] / "logs" / "api.jsonl"

STANDARD_LOG_FIELDS = {
    "name",
    "msg",
    "args",
    "levelname",
    "levelno",
    "pathname",
    "filename",
    "module",
    "exc_info",
    "exc_text",
    "stack_info",
    "lineno",
    "funcName",
    "created",
    "msecs",
    "relativeCreated",
    "thread",
    "threadName",
    "processName",
    "process",
    "message",
    "asctime",
    "taskName",
}

COLOR_RESET = "\033[0m"
LEVEL_COLORS = {
    "DEBUG": "\033[36m",
    "INFO": "\033[34m",
    "WARNING": "\033[33m",
    "ERROR": "\033[31m",
    "CRITICAL": "\033[41m",
}
MESSAGE_COLORS = {
    "DEBUG": "\033[96m",
    "INFO": "\033[94m",
    "WARNING": "\033[93m",
    "ERROR": "\033[91m",
    "CRITICAL": "\033[97m",
}


def ensure_log_dir_exists() -> None:
    LOG_PATH.parent.mkdir(exist_ok = True, parents = True)


class ColorFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        level_color = LEVEL_COLORS.get(record.levelname, COLOR_RESET)
        message_color = MESSAGE_COLORS.get(record.levelname, COLOR_RESET)
        original_levelname = record.levelname
        original_msg = record.msg

        try:
            record.levelname = f"{level_color}{record.levelname}{COLOR_RESET}"
            record.msg = f"{message_color}{record.msg}{COLOR_RESET}"
            return super().format(record)
        except UnicodeEncodeError:
            record.msg = f"{message_color}{str(record.msg).encode('ascii', 'replace').decode('ascii')}{COLOR_RESET}"
            return super().format(record)
        finally:
            record.levelname = original_levelname
            record.msg = original_msg


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "event": getattr(record, "event", None) or record.getMessage().split(" ", 1)[0],
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        extras = {
            key: value
            for key, value in record.__dict__.items()
            if key not in STANDARD_LOG_FIELDS and not key.startswith("_")
        }
        payload.update(extras)
        if record.exc_info:
            payload["error"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def get_console_handler() -> logging.Handler:
    console_handler = logging.StreamHandler()
    try:
        console_handler.stream.reconfigure(encoding = "utf-8")
    except AttributeError:
        if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
            sys.stdout = open(sys.stdout.fileno(), mode = "w", encoding = "utf-8", buffering = 1)

    console_handler.setFormatter(
        ColorFormatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    return console_handler


def get_file_handler() -> logging.Handler:
    file_handler = logging.FileHandler(LOG_PATH, encoding = "utf-8")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    return file_handler


def get_json_file_handler() -> logging.Handler:
    file_handler = logging.FileHandler(JSON_LOG_PATH, encoding = "utf-8")
    file_handler.setFormatter(JsonFormatter())
    return file_handler


def suppress_loggers() -> None:
    for name in (
        "python_multipart",
        "multipart",
        "pdfminer",
        "pdfplumber",
        "httpcore",
        "httpx",
        "uvicorn",
        "asyncio",
        "grpc",
        "aioice",
        "aiortc",
    ):
        logging.getLogger(name).setLevel(logging.WARNING)


def _has_file_handler(root_logger: logging.Logger) -> bool:
    return _has_path_handler(root_logger, LOG_PATH)


def _has_json_file_handler(root_logger: logging.Logger) -> bool:
    return _has_path_handler(root_logger, JSON_LOG_PATH)


def _has_path_handler(root_logger: logging.Logger, path: Path) -> bool:
    log_path = path.resolve()
    for handler in root_logger.handlers:
        base_filename = getattr(handler, "baseFilename", None)
        if not base_filename:
            continue
        if Path(base_filename).resolve() == log_path:
            return True
    return False


def _has_console_handler(root_logger: logging.Logger) -> bool:
    return any(
        isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler)
        for handler in root_logger.handlers
    )


def setup_logging() -> None:
    ensure_log_dir_exists()
    suppress_loggers()

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    if not _has_console_handler(root_logger):
        root_logger.addHandler(get_console_handler())

    if not _has_file_handler(root_logger):
        root_logger.addHandler(get_file_handler())

    if not _has_json_file_handler(root_logger):
        root_logger.addHandler(get_json_file_handler())


logger = logging.getLogger()
