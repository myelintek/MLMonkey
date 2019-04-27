from __future__ import absolute_import

import logging
import logging.handlers
import sys

from mlmonkey import constants

DATE_FORMAT = '%Y-%m-%d %H:%M:%S'


def setup_logging():
    formatter = logging.Formatter(
        fmt="%(asctime)s%(job_id)s [%(levelname)-5s] %(message)s",
        datefmt=DATE_FORMAT,
    )

    log_level = constants.LOG_LEVEL
    logfile_filename = constants.LOG_FILENAME

    # main logger
    main_logger = logging.getLogger('mlmonkey')
    main_logger.setLevel(log_level)

    # Log to stdout
    stdoutHandler = logging.StreamHandler(sys.stdout)
    stdoutHandler.setFormatter(formatter)
    stdoutHandler.setLevel(log_level)
    main_logger.addHandler(stdoutHandler)

    # Log to file
    fileHandler = logging.FileHandler(logfile_filename)
    fileHandler.setFormatter(formatter)
    fileHandler.setLevel(log_level)
    main_logger.addHandler(fileHandler)

    return main_logger


# Do it when this module is loaded
logger = setup_logging()
