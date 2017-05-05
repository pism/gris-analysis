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
                    help="Name of the output shape file", default='interface.shp')



options = parser.parse_args()
filename = options.FILE[0]

driver = ogr.GetDriverByName('ESRI Shapefile')
ds = driver.Open(filename)
layer = ds.GetLayer()
cnt = layer.GetFeatureCount()
last_feature = layer.GetFeature(cnt-1)
nt = last_feature.GetField('timestep')
feature_length_vector = np.zeros(nt)
for feature in layer:
    k = feature.GetField('timestep')
    geom = feature.GetGeometryRef()
    length = geom.GetArea() / 2.
    feature_length_vector[k-1] =+ length

del ds

import pylab as plt
plt.plot(feature_length_vector/1e3)
plt.show()
