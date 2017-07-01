#!/usr/bin/env python
# Copyright (C) 2017 Andy Aschwanden

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
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

from netCDF4 import Dataset as NC

# set up the option parser
parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
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
formatter = logging.Formatter('%(module)s - %(message)s')

# add formatter to ch and fh
ch.setFormatter(formatter)
fh.setFormatter(formatter)

# add ch to logger
logger.addHandler(ch)
logger.addHandler(fh)

thickness_threshold = 10
gdal_gtiff_options = gdal.TranslateOptions(format='GTiff', outputSRS='EPSG:3413')

# Process experiments
dirs = []
dir_gl = 'processed_grounding_lines'
dirs.append(dir_gl)
dir_io = 'processed_ice_ocean'
dirs.append(dir_io)
for dir_processed in dirs:
    if not os.path.isdir(os.path.join(idir, dir_processed)):
        os.mkdir(os.path.join(idir, dir_processed))


exp_files = glob(os.path.join(idir, 'spatial', '*.nc'))
for exp_file in exp_files:
    logger.info('Processing file {}'.format(exp_file))
    exp_basename =  os.path.split(exp_file)[-1].split('.nc')[0]
    logger.info('extracting grounding line')
    exp_gl_wd =  os.path.join(idir, dir_gl, exp_basename + '.shp')
    cmd = ['extract_interface.py', '-t', 'grounding_line', '-o', exp_gl_wd, exp_file]
    sub.call(cmd)
    logger.info('extracting ice ocean interface')
    exp_io_wd =  os.path.join(idir, dir_io, exp_basename + '.shp')
    cmd = ['extract_interface.py', '-t', 'ice_ocean', '-o', exp_io_wd, exp_file]
    sub.call(cmd)
    # logger.info('masking variables where ice thickness < 10m')
    # nco.ncks(input=exp_file, output=exp_nc_wd, variable=','.join([x for x in pvars]), overwrite=True)
    # nco.ncap2(input='-6 -s "{}" {}'.format(ncap2_str, exp_nc_wd), output=exp_nc_wd, overwrite=True)
    # opt = [c.Atted(mode="o", att_name="_FillValue", var_name=myvar, value=fill_value) for myvar in ppvars]
    # nco.ncatted(input=exp_nc_wd, options=opt)
    # for mvar in pvars:
    #     m_exp_nc_wd = 'NETCDF:{}:{}'.format(exp_nc_wd, mvar)
    #     m_exp_gtiff_wd = os.path.join(idir, dir_gtiff, mvar + '_' + exp_basename + '.tif')
    #     logger.info('Converting variable {} to GTiff and save as {}'.format(mvar, m_exp_gtiff_wd))
    #     gdal.Translate(m_exp_gtiff_wd, m_exp_nc_wd, options=gdal_gtiff_options)
    #     if mvar == 'usurf':
    #         m_exp_hs_wd = os.path.join(idir, dir_hs, mvar + '_' + exp_basename + '_hs.tif')
    #         logger.info('Generating hillshade {}'.format(m_exp_hs_wd))
    #         gdal.DEMProcessing(m_exp_hs_wd, m_exp_nc_wd, 'hillshade')
