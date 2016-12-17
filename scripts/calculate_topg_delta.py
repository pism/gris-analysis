#!/usr/bin/env python
# Copyright (C) 2016 Andy Aschwanden


from argparse import ArgumentParser
import os
from nco import Nco
from nco import custom as c
nco = Nco()

import logging
import logging.handlers


# create logger
logger = logging.getLogger('calculate_topg_delta')
logger.setLevel(logging.DEBUG)

# create file handler which logs even debug messages
fh = logging.handlers.RotatingFileHandler('extract.log')
fh.setLevel(logging.DEBUG)
# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
# create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s')

# add formatter to ch and fh
ch.setFormatter(formatter)
fh.setFormatter(formatter)

# add ch to logger
logger.addHandler(ch)
logger.addHandler(fh)


# set up the option parser
parser = ArgumentParser()
parser.description = "Calculate topg-topg_initial."
parser.add_argument("INFILE", nargs=1)
parser.add_argument("OUTFILE", nargs=1)

options = parser.parse_args()

nco.ncap2(input=options.INFILE, output=options.OUTFILE, options='''-O -s "topg_delta=topg-topg_initial;"''')
