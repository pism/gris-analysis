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

options = parser.parse_args()
infile = options.INFILE[0]
outfile = options.OUTFILE[0]
fill_value = -9999

nco.ncks(input=infile, output=outfile, variable='thk', overwrite=True)

nc_in = NC(infile, 'r')
nc_out = NC(outfile, 'a')

z = nc_in.variables['z'][:]
thk = nc_in.variables['thk'][:]
basal_layer_thickness = 200
enthalpy = nc_in.variables['enthalpy'][:, :, :, z<=basal_layer_thickness]
eb_values = np.sum(enthalpy, axis=3)
eb_values[thk<10] = fill_value
eb_var = nc_out.createVariable('basal_enthalpy', 'f', dimensions=('time', 'y', 'x'), fill_value=fill_value)
eb_var.comment = 'enthalpy in the lowermost {}m'.format(basal_layer_thickness)
eb_var.units = 'J kg-1'
eb_var.long_name = 'enthalpy in the lowermost {}m'.format(basal_layer_thickness)
eb_var[:] = eb_values

nc_in.close()
nc_out.close()
