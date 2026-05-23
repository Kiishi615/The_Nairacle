import logging
import sys
import time
from pathlib import Path
from logging.handlers import RotatingFileHandler


def setup_logging(
    level=logging.INFO,
    log_dir="logs",
    max_bytes=10_000_000,
    backup_count=5,
    enable_console=False,
):
    """Configure logging with file + optional console output.

    Args:
        level:          Logging level (10=DEBUG, 20=INFO, 30=WARNING).
        log_dir:        Directory for log files.
        max_bytes:      Max size per log file before rotation.
        backup_count:   Number of rotated log files to keep.
        enable_console: If True, also log to stdout (required for cloud
                        platforms like Koyeb where stdout is the only
                        log target).
    """
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    script_name = Path(sys.argv[0]).stem or "app"
    log_filename = Path(log_dir) / f"{script_name}.log"

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    logger.handlers = []

    # File handler — always enabled for local persistence
    try:
        file_handler = RotatingFileHandler(
            log_filename,
            maxBytes=max_bytes,
            backupCount=backup_count,
        )
        file_handler.setLevel(level)
        logging.Formatter.converter = time.gmtime
        file_format = logging.Formatter(
            "%(asctime)s | %(name)s | %(levelname)s | %(funcName)s | %(message)s"
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
    except (OSError, PermissionError):
        # On read-only filesystems (some cloud platforms), skip file logging
        enable_console = True  # Force console if files aren't writable

    # Console handler — enabled on cloud or when explicitly requested
    if enable_console:
        console_handler = logging.StreamHandler(
            stream=open(sys.stdout.fileno(), "w", encoding="utf-8", closefd=False)
        )
        console_handler.setLevel(level)
        console_format = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        )
        console_handler.setFormatter(console_format)
        logger.addHandler(console_handler)

    # ------------------------------------------------------------------
    # Silence noisy third-party loggers
    # ------------------------------------------------------------------
    for noisy in (
        "httpx", "httpcore", "openai", "anthropic",
        "langchain", "langgraph", "langsmith",
        "pinecone", "tavily", "urllib3",
    ):
        lib_logger = logging.getLogger(noisy)
        lib_logger.setLevel(logging.WARNING)
        lib_logger.propagate = False
        # If the library already added a StreamHandler, remove it
        lib_logger.handlers = [
            h for h in lib_logger.handlers
            if not isinstance(h, logging.StreamHandler)
            or isinstance(h, logging.FileHandler)
        ]

    return str(log_filename)
