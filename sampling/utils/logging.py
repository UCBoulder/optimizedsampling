import logging
import os
import sys

_FORMAT = '[%(asctime)s %(filename)s: %(lineno)3d]: %(message)s'
_LOG_FILE = 'stdout.log'


def setup_logging(cfg):
    logging.root.handlers = []
    logging_config = {'level': logging.INFO, 'format': _FORMAT, 'datefmt': '%Y-%m-%d %H:%M:%S'}
    if cfg.LOG_DEST == 'stdout':
        logging_config['stream'] = sys.stdout
    else:
        log_path = os.path.join(cfg.EXP_DIR, _LOG_FILE)
        if os.path.exists(log_path):
            os.remove(log_path)
        logging_config['filename'] = log_path
    logging.basicConfig(**logging_config)


def get_logger(name):
    return logging.getLogger(name)
