#!/usr/bin/env python
# Copyright (C) 2017, 2018 Andy Aschwanden

import numpy as np
from argparse import ArgumentParser
from netCDF4 import Dataset as NC


if __name__ == "__main__":


    parser = ArgumentParser()
    parser.description = "Calculate percentage of ensemble members that lose percentage of mass."
    parser.add_argument("FILE", nargs='*',
                        help="File to process", default=None)
    parser.add_argument("-t", dest='threshold', type=float,
                        help="Threshold in percent", default=90)
    options = parser.parse_args()

    threshold = options.threshold
    infiles = options.FILE

    var = 'limnsw'

    count = 0
    nf = len(infiles)
    for infile in infiles:
        nc = NC(infile, 'r')

        mass_loss =  np.abs((nc.variables[var][0] - nc.variables[var][-1]) / nc.variables[var][0] * 100)
        if mass_loss >= threshold:
            count += 1

        nc.close()
    pc = float(count) / float(nf) * 100.
    print('{} percent of simulations are below {}'.format(pc, threshold))
