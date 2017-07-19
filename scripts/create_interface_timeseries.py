#!/usr/bin/env python

import numpy as np
from argparse import ArgumentParser
from netCDF4 import Dataset as NC
from netcdftime import utime
import gdal
import ogr
import osr
import os
from pyproj import Proj
import logging
import logging.handlers

try:
    import pypismtools.pypismtools as ppt
except:
    import pypismtools as ppt

# create logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# create file handler which logs even debug messages
fh = logging.handlers.RotatingFileHandler('extract.log')
fh.setLevel(logging.DEBUG)
# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.ERROR)
# create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s')

# add formatter to ch and fh
ch.setFormatter(formatter)
fh.setFormatter(formatter)

# add ch to logger
logger.addHandler(ch)
logger.addHandler(fh)




parser = ArgumentParser(
    description='''A script to extract interfaces (calving front, ice-ocean, or groundling line) from a PISM netCDF file, and save it as a shapefile (polygon).''')
parser.add_argument("FILE", nargs=1)
parser.add_argument("-o", "--output_filename", dest="out_file",
                    help="Name of the output file", default='plot.pdf')

options = parser.parse_args()
filename = options.FILE[0]
ofile = options.out_file

driver = ogr.GetDriverByName('ESRI Shapefile')
ds = driver.Open(filename)
layer = ds.GetLayer()
cnt = layer.GetFeatureCount()
dates = []
t = []
lengths = []

for feature in layer:
    dates.append(feature.GetField('timestamp'))
    t.append(feature.GetField('timestep'))
    geom = feature.GetGeometryRef()
    length = geom.GetArea() / 2.
    lengths.append(length)

del ds

dates = np.array(dates)
t = np.array(t)
lengths = np.array(lengths)

import pylab as plt
plt.plot(t, lengths/1e3, 'o')
plt.savefig(ofile)
