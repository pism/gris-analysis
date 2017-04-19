#!/usr/bin/env python
# Copyright (C) 2017 Andy Aschwanden

import os
try:
    import subprocess32 as sub
except:
    import subprocess as sub
from glob import glob
import gdal
import numpy as np
from nco import Nco
nco = Nco()
from nco import custom as c
import logging
import logging.handlers
from argparse import ArgumentParser

from netCDF4 import Dataset as NC
from netcdftime import utime

def calc_deglaciation_time(infile, thickness_threshold):
    '''
    Calculate year of deglaciation (e.g. when ice thickness
    drops below threshold 'thickness_threshold')
    '''
    nc = NC(infile, 'a')
    time = nc.variables['time']
    time_units = time.units
    time_calendar = time.calendar
    thk = nc.variables['thk'][:]
    x = nc.variables['x'][:]
    y = nc.variables['y'][:]
    deglac_time = nc.createVariable(mvar, 'f', dimensions=('y', 'x'), fill_value=0)
    deglac_time.long_name = 'year of deglaciation'
    nx = len(x)
    ny = len(y)
    for n in  range(ny):
        for m in range(nx):
             try:
                 idx = np.where(thk[:,n,m] < thickness_threshold)[0][0]
                 deglac_time[n,m] = time[idx] / secpera
             except:
                 pass
    nc.close()


# set up the option parser
parser = ArgumentParser()
parser.description = "Postprocessing files."
parser.add_argument("INDIR", nargs=1,
                    help="main directory", default=None)

options = parser.parse_args()
idir = options.INDIR[0]

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

thickness_threshold = 10
secpera = 24 * 3600 * 365  # for 365_day calendar only
gdal_gtiff_options = gdal.TranslateOptions(format='GTiff', outputSRS='EPSG:3413')

# Process experiments
dir_nc = 'processed_spatial_nc'
dir_gtiff = 'processed_spatial_gtiff'
mvar = 'deglac_year'

for dir_processed in (dir_gtiff, dir_nc):
    if not os.path.isdir(os.path.join(idir, dir_processed)):
        os.mkdir(os.path.join(idir, dir_processed))

exp_files = glob(os.path.join(idir, 'spatial', '*.nc'))
for exp_file in exp_files:
    logger.info('Processing file {}'.format(exp_file))
    exp_basename =  os.path.split(exp_file)[-1].split('.nc')[0]
    exp_nc_wd = os.path.join(idir, dir_nc, exp_basename + '.nc')
    nco.ncks(input=exp_file, output=exp_nc_wd, overwrite=True, variable=['thk'])
    calc_deglaciation_time(exp_nc_wd, thickness_threshold)
    m_exp_nc_wd = 'NETCDF:{}:{}'.format(exp_nc_wd, mvar)
    m_exp_gtiff_wd = os.path.join(idir, dir_gtiff, mvar + '_' + exp_basename + '.tif')
    logger.info('Converting variable {} to GTiff and save as {}'.format(mvar, m_exp_gtiff_wd))
    gdal.Translate(m_exp_gtiff_wd, m_exp_nc_wd, options=gdal_gtiff_options)

