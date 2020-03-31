#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Does this and that.

For usage information, call with --help.

Author: Jan Schlüter
"""

import sys
import os
from argparse import ArgumentParser

import numpy as np


def opts_parser():
    usage =\
"""Does this and that.
"""
    parser = ArgumentParser(description=usage)
    parser.add_argument('positional1',  # exactly one
            type=str,
            help='The thing.')
    parser.add_argument('positional2', nargs='+',  # at least one
            type=str,
            help='The other things.')
    parser.add_argument('positional3', nargs='*',  # at least zero
            type=str,
            help='Optional other things.')
    parser.add_argument('-i', '--ixample',
            type=int, default=100,
            help='Asdf (default: %(default)s)')
    parser.add_argument('-b', '--bexample',
            action='store_true', default=False,
            help='If given, foo bar.')
    parser.add_argument('--cexample',
            type=str, choices=('alpha', 'beta'), default='alpha',
            help='Asdf: alpha or beta. (default: %(default)s).')
    parser.add_argument('--lexample',
            action='append', type=float,
            help='Asdf.')
    return parser


def main():
    # parse command line
    parser = opts_parser()
    options = parser.parse_args()
    if False:
        parser.error("complaining about this or that")

    # ...


if __name__ == "__main__":
    main()

