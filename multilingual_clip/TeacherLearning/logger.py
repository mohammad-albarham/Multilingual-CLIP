import logging
from rich.logging import RichHandler


logger = logging.getLogger(__name__)

shell_handler = RichHandler()
file_handler = logging.FileHandler("translation_debug.log")

logger.setLevel(logging.INFO)
shell_handler.setLevel(logging.INFO)
file_handler.setLevel(logging.INFO)

fmt_shell = '%(message)s'
fmt_file = '%(levelname)s %(asctime)s [%(filename)s:%(funcName)s:%(lineno)d] %(message)s'

shell_formatter = logging.Formatter(fmt_shell)
file_formatter = logging.Formatter(fmt_file)

# here we hook everything together
shell_handler.setFormatter(shell_formatter)
file_handler.setFormatter(file_formatter)

logger.addHandler(shell_handler)
logger.addHandler(file_handler)