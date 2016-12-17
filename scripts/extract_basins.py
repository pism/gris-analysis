#!/usr/bin/env python
# Copyright (C) 2016 Andy Aschwanden


from argparse import ArgumentParser
import tempfile
import ocgis
import os
from netcdftime import datetime
from cdo import Cdo
cdo = Cdo()
from nco import Nco
from nco import custom as c
nco = Nco()

import logging
import logging.handlers

try:
    import pypismtools.pypismtools as ppt
except:
    import pypismtools as ppt

# create logger
logger = logging.getLogger('extract_basins')
logger.setLevel(logging.DEBUG)

# create file handler which logs even debug messages
fh = logging.handlers.RotatingFileHandler('extract.log')
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


# set up the option parser
parser = ArgumentParser()
parser.description = "Generating scripts for prognostic simulations."
parser.add_argument("FILE", nargs=1)
parser.add_argument("--shape_file", dest="shape_file",
                    help="Path to shape file with basins", default=None)
parser.add_argument("--o_dir", dest="odir",
                    help="output directory", default='.')

options = parser.parse_args()
URI = options.FILE[0]
SHAPEFILE_PATH = options.shape_file

odir = options.odir
if not os.path.isdir(odir):
    os.mkdir(odir)

ocgis.env.OVERWRITE = True


## the target variable in the dataset to convert
VARIABLE = ['basal_mass_balance_average',
            'eigen_calving_rate',
            'surface_mass_balance_average',
            'vonmises_calving_rate']

# Output name
savename=URI[0:len(URI)-3] 

## set the output format to convert to
output_format = 'nc'

## we can either subset the data by a geometry from a shapefile, or convert to
## geojson for the entire spatial domain. there are other options here (i.e. a
## bounding box for tiling or a Shapely geometry).
GEOM = SHAPEFILE_PATH

basins = range(1, 9)

# date_start = '-11700-1-1'
# date_end = '1-1-1'
# r_date_start = date_start.split('-')
# r_date_end = date_end.split('-')
# print r_date_start, r_date_end
# time_range = [datetime(int(r_date_start[0]), int(r_date_start[1]), int(r_date_start[2])),
#               datetime(int(r_date_end[0]), int(r_date_end[1]), int(r_date_end[2]))]
# rd = ocgis.RequestDataset(uri=URI,variable=VARIABLE,time_range=time_range)

rd = ocgis.RequestDataset(uri=URI, variable=VARIABLE)
for basin in basins:
    logger.info('Extracting basin {}'.format(basin))
    if GEOM is None:
        select_ugid = None
    else:
        select_geom = filter(lambda x: x['properties']['basin'] == basin,
                             ocgis.GeomCabinetIterator(path=SHAPEFILE_PATH))
        ## this argument must always come in as a list
        select_ugid = [select_geom[0]['properties']['UGID']]
    prefix = 'basin_{basin}_{savename}'.format(basin=basin, savename=savename)
    ## parameterize the operations to be performed on the target dataset
    ops = ocgis.OcgOperations(dataset=rd,
                              geom=SHAPEFILE_PATH,
                              aggregate=False,
                              snippet=False,
                              select_ugid=select_ugid,
                              output_format=output_format,
                              prefix=prefix,
                              dir_output=odir)
    ret = ops.execute()
    print('path to output file: {0}'.format(ret))
    output = os.path.join(odir, prefix, '.'.join(['_'.join(['scalar', prefix]), 'nc']))
    logger.info('Calculating field sum and saving to \n {}'.format(output))
    cdo.fldsum(input=ret, output=output)
    logger.info('Updating units in \n {}'.format(output))
    opt = [c.Atted(mode="o", att_name="units", var_name="surface_mass_balance_average", value="kg year-1"),
           c.Atted(mode="o", att_name="units", var_name="basal_mass_balance_average", value="kg year-1")]
    nco.ncatted(input=output, options=opt)
    #cdo.expr('ratio=surface_mass_balance_average/basal_mass_balance_average', input=output, output=output)

