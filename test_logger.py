from . import logger

log = logger.get_logger("test_logger")

log.info("This is an info message.")
log.warning("This is a warning message.")
log.error("This is an error message.")
log.debug("This is a debug message.")
log.critical("This is a critical message.")