import logging
import logging.handlers
import os
from pathlib import Path
from datetime import datetime
from typing import Optional

from .config import settings


# Create logs directory if it doesn't exist
def setup_logging() -> logging.Logger:
    """
    Configure and setup logging for the application.
    
    Returns:
        logging.Logger: Configured logger instance
    """
    # Create logs directory
    if settings.LOG_FILE:
        log_dir = Path(settings.LOG_FILE).parent
        log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger(settings.APP_NAME)
    logger.setLevel(getattr(logging, settings.LOG_LEVEL))
    
    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        fmt=settings.LOG_FORMAT,
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Console handler (always enabled)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, settings.LOG_LEVEL))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if LOG_FILE is configured)
    if settings.LOG_FILE:
        try:
            # Use rotating file handler to manage log file size
            file_handler = logging.handlers.RotatingFileHandler(
                filename=settings.LOG_FILE,
                maxBytes=10 * 1024 * 1024,  # 10MB
                backupCount=5,
                encoding="utf-8"
            )
            file_handler.setLevel(getattr(logging, settings.LOG_LEVEL))
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except Exception as e:
            logger.warning(f"Failed to setup file logging: {e}")
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    return logger


# Initialize logger
logger = setup_logging()


def get_logger(module_name: Optional[str] = None) -> logging.Logger:
    """
    Get a logger instance for a specific module.
    
    Args:
        module_name: Name of the module (typically __name__)
    
    Returns:
        logging.Logger: Logger instance for the module
    """
    if module_name:
        return logging.getLogger(f"{settings.APP_NAME}.{module_name}")
    return logger


# Convenience functions
def log_startup_info():
    """Log application startup information"""
    logger.info("=" * 60)
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    logger.info(f"Debug Mode: {settings.DEBUG}")
    logger.info(f"LLM Model: {settings.LLM_MODEL_NAME}")
    logger.info(f"LLM Type: {settings.LLM_TYPE}")
    logger.info(f"Vector Store: {settings.VECTOR_STORE_TYPE}")
    logger.info(f"Log Level: {settings.LOG_LEVEL}")
    logger.info("=" * 60)


def log_shutdown_info():
    """Log application shutdown information"""
    logger.info("=" * 60)
    logger.info(f"Shutting down {settings.APP_NAME}")
    logger.info("=" * 60)
