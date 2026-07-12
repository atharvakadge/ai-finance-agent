"""
Logging configuration for the application.

Why proper logging matters:
- print() disappears when running as a service (no terminal)
- Logs can be sent to files, monitoring tools (Datadog, CloudWatch)
- Log LEVELS let you filter: DEBUG in dev, WARNING in prod
- Timestamps + context help debug issues hours/days later

Levels (in order):
- DEBUG: detailed dev info ("embedding 1120 chunks")
- INFO: normal operations ("server started", "PDF uploaded")
- WARNING: something unexpected but not broken ("slow API response")
- ERROR: something failed ("LLM API call failed")
- CRITICAL: app is broken ("cannot connect to vector DB")
"""

import logging
import sys


def setup_logging(level: str = "INFO") -> logging.Logger:
    """
    Configure logging for the entire application.

    Returns a logger that all modules should use.
    """
    # Create a formatter with timestamp, level, module, and message
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler — logs go to terminal
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    # Root logger — all loggers inherit from this
    root_logger = logging.getLogger("finlens")
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    root_logger.addHandler(console_handler)

    # Reduce noise from third-party libraries
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    return root_logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for a specific module.

    Usage in any file:
        from app.logging_config import get_logger
        logger = get_logger(__name__)
        logger.info("Something happened")
    """
    return logging.getLogger(f"finlens.{name}")