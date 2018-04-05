#!/usr/bin/env python
# Copyright (C) 2017 Andy Aschwanden

from argparse import ArgumentParser
from netCDF4 import Dataset as NC
import numpy as np

from nco import Nco
nco = Nco()

# Set up the option parser
parser = ArgumentParser()
parser.description = "Extract basal enthalpy."
parser.add_argument("INFILE", nargs=1)
parser.add_argument("OUTFILE", nargs=1)
parser.add_argument("-t", "--thickness_threshold", dest="basal_layer_thickness_percent",
                    help="Percent ice thickness used for basal layer thickness", default=10)


options = parser.parse_args()
infile = options.INFILE[0]
outfile = options.OUTFILE[0]
basal_layer_thickness_percent = options.basal_layer_thickness_percent
fill_value = -9999

nco.ncks(input=infile, output=outfile, variable='thk', overwrite=True)

nc_in = NC(infile, 'r')
nc_out = NC(outfile, 'a')
eb_var = nc_out.createVariable('basal_enthalpy', 'f', dimensions=('time', 'y', 'x'), fill_value=fill_value)

z = nc_in.variables['z'][:]
thk = nc_in.variables['thk'][:]
n_times, n_rows, n_cols = nc_in.variables['thk'].shape
eb_values = np.zeros_like(thk)
for t in range(n_times):
    for r in  range(n_rows):
        for c in range(n_cols):
            print('{},{},{}'.format(t, r, c))
            thickness_ij = thk[t, r, c]
            enthalpy = nc_in.variables['enthalpy'][t, r, c, z <= basal_layer_thickness_percent * thickness_ij]
            eb_values[t, r, c] = np.sum(enthalpy)
            eb_values[thk<10] = fill_value
eb_var.comment = 'enthalpy in the lowermost {}%'.format(basal_layer_thickness_percent)
eb_var.units = 'J kg-1'
eb_var.long_name = 'enthalpy in the lowermost {}%'.format(basal_layer_thickness_percent)
eb_var[:] = eb_values

nc_in.close()
nc_out.close()
