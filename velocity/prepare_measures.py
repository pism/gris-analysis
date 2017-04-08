#!/usr/bin/env python
# Copyright (C) 2017 Andy Aschwanden

import os
try:
    import subprocess32 as sub
except:
    import subprocess as sub
import gdal
from nco import Nco
nco = Nco()
from nco import custom as c

import logging
import logging.handlers

# create logger
logger = logging.getLogger('prepare_measures')
logger.setLevel(logging.DEBUG)

# create file handler which logs even debug messages
fh = logging.handlers.RotatingFileHandler('prepare_measures.log')
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

var_dict = {'vx': 'uvelsurf', 'vy': 'vvelsurf', 'ex': 'uvelsurf_error', 'ey': 'vvelsurf_error'}

basedir = 'measures'
basename = 'greenland_vel_mosaic250'
version = 'v1'
gdal_nc_options = gdal.TranslateOptions(format='netCDF')

ifile = '.'.join(['_'.join([basename, version]), 'tif'])
ofile_merged = '.'.join(['_'.join([basename, version]), 'nc'])
ifile_wp = os.path.join(basedir, ifile)
ofile_merged_wp = os.path.join(basedir, ofile_merged) 
logger.info('Converting {} to {}'.format(ifile_wp, ofile_merged_wp))
gdal.Translate(ofile_merged_wp, ifile_wp, options=gdal_nc_options)
rDict = {'Band1': 'velsurf_mag'}
opt = [c.Rename('variable', rDict)]
nco.ncrename(input=ofile_merged_wp, options=opt)
opt = [c.Atted(mode="o", att_name="proj4", var_name='global', value="+init=epsg:3413", stype='c'),
        c.Atted(mode="o", att_name="units", var_name='velsurf_mag', value="m year-1")]
nco.ncatted(input=ofile_merged_wp, options=opt)
for mvar in ('vx', 'vy', 'ex', 'ey'):
    ifile = '.'.join(['_'.join([basename, mvar, version]), 'tif'])
    ofile = '.'.join(['_'.join([basename, mvar, version]), 'nc'])
    ofile_wp = os.path.join(basedir, ofile)
    ifile_wp = os.path.join(basedir, ifile)
    gdal.Translate(ofile_wp, ifile_wp, options=gdal_nc_options)
    rDict = {'Band1': var_dict[mvar]}
    opt = [c.Rename('variable', rDict)]
    nco.ncrename(input=ofile_wp, options=opt)
    opt = [c.Atted(mode="o", att_name="units", var_name=var_dict[mvar], value="m year-1")]
    nco.ncatted(input=ofile_wp, options=opt)
    nco.ncks(input=ofile_wp, output=ofile_merged_wp, append=True)
ncap2_str = 'velsurf_mag_error=sqrt(uvelsurf_error^2+vvelsurf_error^2); velsurf_normal_error=velsurf_mag_error;'
nco.ncap2(input='-s "{}" {}'.format(ncap2_str, ofile_merged_wp), output=ofile_merged_wp, overwrite=True)
