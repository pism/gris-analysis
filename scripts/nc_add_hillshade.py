#!/usr/bin/env python
# Copyright (C) 2017 Andy Aschwanden

import os
try:
    import subprocess32 as sub
except:
    import subprocess as sub
from glob import glob
import numpy as np
import gdal
from nco import Nco
nco = Nco()
from nco import custom as c
import logging
import logging.handlers
from argparse import ArgumentParser

from netCDF4 import Dataset as NC

# set up the option parser
parser = ArgumentParser()
parser.description = "Postprocessing files."
parser.add_argument("FILE", nargs=1,
                    help="file", default=None)

options = parser.parse_args()
exp_file = options.FILE[0]

# create logger
logger = logging.getLogger('postprocess')
logger.setLevel(logging.DEBUG)

# create file handler which logs even debug messages
fh = logging.handlers.RotatingFileHandler('prepare_velocity_observations.log')
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

def cart2pol(x, y):
    '''
    cartesian to polar coordinates
    '''
    theta = np.arctan2(y, x)
    rho = np.sqrt(x**2 + y**2)
    return (theta, rho) 

def hillshade(dem, dx, dy, azimuth, altitude, zf):
   '''
   shaded relief using the ESRI algorithm
   '''
   # lighting azimuth
   azimuth = 360.0-azimuth+90 #convert to mathematic unit
   if (azimuth>360) or (azimuth==360):
      azimuth = azimuth - 360
   azimuth = azimuth * (np.pi / 180)  # convert to radians

   # lighting altitude
   altitude = (90 - altitude) * (np.pi / 180)  # convert to zenith angle in radians

   # calc slope and aspect (radians)
   fx, fy = np.gradient(dem, dx)  # uses simple, unweighted gradient of immediate
   [asp, grad] = cart2pol(fy, fx)  # % convert to carthesian coordinates

   grad = np.arctan(zf * grad) #steepest slope
   # convert asp
   asp[asp<np.pi] = asp[asp < np.pi]+(np.pi / 2)
   asp[asp<0] = asp[asp<0] + (2 * np.pi)

   ## hillshade calculation
   h = 255.0 * ((np.cos(altitude) * np.cos(grad))
                + (np.sin(altitude) * np.sin(grad) * np.cos(azimuth - asp)))
   h[h<0] = 0 # % set hillshade values to min of 0.

   return h




azimuth = 45
zf = 5
altitude = 45

dx = dy = 1500

fill_value = 0
ncap2_str = 'usurf_hs=usurf*0;'
ppvars = ['usurf_hs']

logger.info('Processing file {}'.format(exp_file))
#nco.ncap2(input='-4 -s "{}" {}'.format(ncap2_str, exp_file), output=exp_file, overwrite=True)
opt = [c.Atted(mode="o", att_name="_FillValue", var_name=myvar, value=fill_value) for myvar in ppvars]
#nco.ncatted(input=exp_file, options=opt)

nc = NC(exp_file, 'a')
nt = len(nc.variables['time'][:])
usurf = nc.variables['usurf'][:]
thk = nc.variables['thk'][:]
speed = nc.variables['velsurf_mag'][:]
if 'usurf_hs' not in nc.variables:
    nc.createVariable('usurf_hs', 'i', dimensions=('time', 'y', 'x'), fill_value=fill_value)
for t in range(nt):
    print t
    dem = usurf[t, Ellipsis]
    mthk = thk[t, Ellipsis]
    mspeed = speed[t, Ellipsis]
    hs = hillshade(dem, dx, dy, azimuth, altitude, zf)
    hs[dem==0] = fill_value
    hs[mthk<=10] = fill_value
    mspeed[mthk<=10] = -2e9
    nc.variables['usurf_hs'][t, Ellipsis] = hs
    nc.variables['velsurf_mag'][t, Ellipsis] = mspeed

nc.close()
