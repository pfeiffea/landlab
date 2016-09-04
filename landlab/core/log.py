#! /usr/bin/env python
"""Issue landlab logging messages.
"""
from __future__ import print_function

import os
import sys
import textwrap
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


wrapper = textwrap.TextWrapper()


def log_level():
    """Get the current log level.
    
    Returns
    -------
    int
        The current log level.
    """
    return int(os.environ.get('LANDLAB_LOG_LEVEL', WARNING))


def paragraphs(msg):
    """Iterate over paragraphs of a string.

    Parameters
    ----------
    msg : str
        The string to split into paragraphs.

    Yields
    ------
    str
        A paragraph.
    """
    lines = msg.splitlines(True)
    for group_separator, line_iteration in groupby(lines, key=str.isspace):
        if not group_separator:
            yield ''.join(line_iteration)


def dedent_paragraph(para):
    """Remove common leading whitespace from a paragraph.

    Parameters
    ----------
    para : str
        A paragraph.

    Returns
    -------
    str
        The dedented paragraph.
    """
    lines = para.splitlines()
    dedented = os.linesep.join([
        lines[0].strip(),
        textwrap.dedent(os.linesep.join(lines[1:]))
    ])
    return wrapper.fill(dedented)


def pretty_paragraphs(msg):
    """Make paragraphs look pretty.

    Parameters
    ----------
    msg : str
        String containing paragraphs.

    Returns
    -------
    list or str
        A list of the paragraph strings.
    """
    return [dedent_paragraph(p) for p in paragraphs(msg)]


def prettify_message(lvl, msg, long=None):
    """Make a log message pretty.

    Parameters
    ----------
    lvl : int
        Log level.
    msg : str
        The short log message.
    long : str, optional
        A longer message to print after the brief message.

    Returns
    -------
    str
        The pretty log message.
    """
    try:
        prefix = PREFIX[lvl]
    except KeyError:
        raise ValueError('log level not understood')

    short = '{prefix}: {msg}'.format(prefix=PREFIX[lvl], msg=msg)
    if long:
        blankline = os.linesep * 2
        return blankline.join([short] + pretty_paragraphs(long))
    else:
        return short


def log(lvl, msg, long=None):
    """Print a message with a log level.

    Parameters
    ----------
    lvl : int
        Log level.
    msg : str
        The short log message.
    long : str, optional
        A longer message to print after the brief message.
    """
    if lvl >= log_level():
        print(prettify_message(lvl, msg, long=long), file=sys.stderr)


def critical(msg, long=None):
    return log(CRITICAL, msg, long=long)


def error(msg, long=None):
    return log(ERROR, msg, long=long)


def warning(msg, long=None):
    return log(WARNING, msg, long=long)


def info(msg, long=None):
    return log(INFO, msg, long=long)


def debug(msg, long=None):
    return log(DEBUG, msg, long=long)
