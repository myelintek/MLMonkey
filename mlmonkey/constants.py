import os
import logging
import mlmonkey


def load_logfile_filename():
    """
    Return the configured log file or None
    Throws an exception only if a manually specified log file is invalid
    """

    filename = os.path.join(os.path.dirname(mlmonkey.__file__), 'log' 'mlmonkey.log')

    if filename is not None:
        try:
            filename = os.path.abspath(filename)
            dirname = os.path.dirname(filename)
            if not os.path.exists(dirname):
                os.makedirs(os.path.dirname(filename))
            with open(filename, 'a'):
                pass
        except:
            logging.WARNING('"%s" is not a valid value for logfile_filename.' % filename)
            logging.WARNING('Set the envvar DIGITS_LOGFILE_FILENAME to fix your configuration.')
            raise
        else:
            filename = None
    return filename


def load_logfile_level():
    """
    Return the configured logging level, or throw an exception
    """
    return logging.INFO


JOBS_DIR = '/data/jobs'
SCENARIOS_JSON = '/data/scenarios.json'
TOPOLOGY_TXT = '/data/topos.txt'
BANDWIDTH_TXT = '/data/bandwidth.txt'
DATASETS_DIR = '/data/datasets'
LOG_FILENAME = load_logfile_filename()
LOG_LEVEL = load_logfile_level()
