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
                    type=float, help="Percent ice thickness used for basal layer thickness", default=10)


options = parser.parse_args()
infile = options.INFILE[0]
outfile = options.OUTFILE[0]
basal_layer_thickness_fraction = options.basal_layer_thickness_percent / 100.0
fill_value = -9999
thk_min = 10.0

print("Creating the output file...")
nco.ncks(input=infile, output=outfile, variable='thk', overwrite=True)
print("done")

def process_block(t, block, z, thk, basal_layer_thickness_fraction, result):
    n_rows, n_cols, n_z = block.shape
    for r in  range(n_rows):
        for c in range(n_cols):
            thickness_ij = thk[t, r, c]

            # skip ice-free points
            if thickness_ij < thk_min:
                continue

            z_max = basal_layer_thickness_fraction * thickness_ij
            S = 0.0
            for k in range(n_z):
                if z[k] > z_max:
                    if k > 0 and z[k-1] <= z_max:
                        delta = (z_max - z[k-1]) / (z[k] - z[k-1])
                        S += block[r, c, k] * delta
                    break
                S += block[r, c, k]

            result[t, r, c] += S

nc_in = NC(infile, 'r')
z = nc_in.variables['z'][:]
Mz = len(z)
thk = nc_in.variables['thk'][:]
enthalpy = nc_in.variables['enthalpy']

# dimensions: time, y, x, z; z is fourth
Nz = enthalpy.chunking()[3]

nc_out = NC(outfile, 'a')
eb_var = nc_out.createVariable('basal_enthalpy', 'f', dimensions=('time', 'y', 'x'), fill_value=fill_value)

n_times, n_rows, n_cols = thk.shape
eb_values = np.zeros_like(thk)
for t in range(n_times):
    k = 0
    while k + Nz < Mz + 1:
        print("Processing record {}, block {}:{}...".format(t, k, k + Nz - 1))
        data = enthalpy[t, :, :, k:k + Nz]
        process_block(t, data, z[k:k + Nz], thk, basal_layer_thickness_fraction, eb_values)
        k += Nz

    if k < Mz:
        print("Processing record {}, the last block...".format(t))
        data = enthalpy[t, :, :, k:]
        process_block(t, data, z[k:], thk, basal_layer_thickness_fraction, eb_values)

eb_values[thk < thk_min] = fill_value

eb_var.units = 'J kg-1'
eb_var.long_name = 'enthalpy in the lowermost {}%'.format(options.basal_layer_thickness_percent)
eb_var.comment = eb_var.long_name
eb_var[:] = eb_values

nc_in.close()
nc_out.close()
