#! /usr/bin/env python
"""Issue landlab logging messages.
"""
from __future__ import print_function

import os
import sys
from textwrap import TextWrapper
from itertools import groupby


CRITICAL = 50
ERROR = 40
WARNING = 30
INFO = 20
DEBUG = 10
NOTSET = 0

PREFIX = {
    CRITICAL: 'CRITICAL',
    ERROR: 'ERROR',
    WARNING: 'WARNING',
    INFO: 'INFO',
    DEBUG: 'DEBUG',
    NOTSET: 'NOTSET',
}


wrapper = TextWrapper()


def log_level():
    """Get the current log level.
    
    Returns
    -------
    int
        The current log level.
    """
    return int(os.environ.get('LANDLAB_LOG_LEVEL', WARNING))


def paragraphs(msg):
    """Split a string into paragraphs.

    Parameters
    ----------
    msg : str
        The string to split.

    Yields
    ------
    str
        A paragraph.
    """
    lines = msg.splitlines(True)
    for group_separator, line_iteration in groupby(lines, key=str.isspace):
        if not group_separator:
            yield ''.join(line_iteration)


def prettify_message(lvl, msg, long=None):
    try:
        prefix = PREFIX[lvl]
    except KeyError:
        raise ValueError('log level not understood')
    short = '{prefix}: {msg}'.format(prefix=PREFIX[lvl], msg=msg)
    if long:
        blankline = os.linesep * 2
        return blankline.join([short] +
                              [wrapper.fill(p) for p in paragraphs(long)])
    else:
        return short


def log(lvl, msg, long=None):
    if lvl >= log_level():
        print(prettify_message(lvl, msg, long=long), file=sys.stderr)


def critical(msg, long=None):
    return log(CRITICAL, msg, long=long)


def error(msg, long=None):
    return log(ERROR, msg, long=long)


def warn(msg, long=None):
    return log(WARNING, msg, long=long)


def info(msg, long=None):
    return log(INFO, msg, long=long)


def debug(msg, long=None):
    return log(DEBUG, msg, long=long)
