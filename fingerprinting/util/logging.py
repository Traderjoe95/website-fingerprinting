import atexit
import multiprocessing
import sys
from logging import DEBUG, INFO, WARN, ERROR, CRITICAL, Formatter, addLevelName, StreamHandler, getLogger, \
    Handler
from logging.handlers import QueueHandler, QueueListener
from queue import Queue
from typing import Optional

import progressbar

from .config import CONFIG

__queue: Optional[Queue] = None
__listener: Optional[QueueListener] = None


def __level(name: str) -> int:
    lower_name = name.lower()

    if lower_name == "debug":
        return DEBUG
    elif lower_name == "info":
        return INFO
    elif lower_name == "warn" or lower_name == "warning":
        return WARN
    elif lower_name == "error":
        return ERROR
    elif lower_name == "fatal" or lower_name == "critical":
        return CRITICAL

    raise ValueError(f"Unrecognized log level '{name}'")


def configure_worker(q: Queue, override_level: Optional[int] = None):
    global __queue

    addLevelName(WARN, "WARN")
    addLevelName(CRITICAL, "FATAL")

    log_level = __level(CONFIG.get_str("logging.level", "INFO"))

    if override_level is not None:
        log_level = min(log_level, override_level)

    __queue = q

    root = getLogger()

    if not any(isinstance(h, QueueHandler) for h in root.handlers):
        root.addHandler(get_handler())

    root.setLevel(log_level)


def configure(override_level: Optional[int] = None):
    global __listener, __queue

    if __listener is not None:
        raise RuntimeError("The logger was already configured")

    progressbar.streams.wrap_stderr()
    progressbar.streams.wrap_stdout()

    configure_worker(multiprocessing.Manager().Queue(-1), override_level)

    fmt = '%(asctime)s %(levelname)-5s - %(name)45s:%(lineno)-4d - %(message)s'

    Formatter.default_msec_format = '%s.%03d'

    handler = StreamHandler(sys.stdout)
    handler.setFormatter(Formatter(fmt))

    __listener = QueueListener(__queue, handler)
    __listener.start()

    atexit.register(__listener.stop)


def get_queue() -> multiprocessing.Queue:
    return __queue


def get_handler(q: Optional[multiprocessing.Queue] = None) -> Handler:
    if q is None and __queue is None:
        raise ValueError("No queue defined")

    return QueueHandler(__queue if q is None else q)
