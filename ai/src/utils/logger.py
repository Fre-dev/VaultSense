import logging
import os
from colorlog import ColoredFormatter


def setup_logger(name):
    # Create a logger
    logger = logging.getLogger(name)
    logger.setLevel(os.getenv("LOG_LEVEL"))

    # Create a console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(os.getenv("LOG_LEVEL"))

    # Define the color format
    formatter = ColoredFormatter(
        "%(log_color)s%(levelname)-8s%(reset)s %(blue)s%(name)s:%(lineno)d - %(message)s",
        datefmt=None,
        reset=True,
        log_colors={
            "DEBUG": "cyan",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "red,bg_white",
        },
        secondary_log_colors={},
        style="%",
    )

    # Add the formatter to the handler
    console_handler.setFormatter(formatter)

    # Add the handler to the logger
    if not logger.hasHandlers():
        logger.addHandler(console_handler)

    return logger
