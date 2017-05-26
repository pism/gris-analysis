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

try:
    import subprocess32 as sub
except:
    import subprocess as sub

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
parser.add_argument("--o_dir", dest="odir",
                    help="output directory", default='.')
parser.add_argument("--shape_file", dest="shape_file",
                    help="Path to shape file with basins", default=None)
parser.add_argument("-v", "--variable", dest="VARIABLE",
                    help="Comma-separated list of variables to be extracted. By default, all variables are extracted.", default=None)

options = parser.parse_args()
URI = options.FILE[0]
SHAPEFILE_PATH = options.shape_file
if options.VARIABLE is not None:
    VARIABLE=options.VARIABLE.split(',')
else:
    VARIABLE=options.VARIABLE

odir = options.odir
if not os.path.isdir(odir):
    os.mkdir(odir)

# VARIABLE='beta,cell_area,dHdt,dbdt,discharge_mass_flux,flux_divergence,height_above_flotation,ice_mass,sftgif,tauc,taub_mag,temppabase,tempsurf,thk,topg,usurf,uvelbase,vvelbase,velbase_mag,uvelsurf,vvelsurf,velsurf_mag'.split(',')
# VARIABLE='beta,cell_area,dHdt,discharge_mass_flux,flux_divergence,height_above_flotation,ice_mass,sftgif,temppabase,thk,topg,usurf,velbase_mag,velsurf_mag'.split(',')
# print VARIABLE
ocgis.env.OVERWRITE = True

# Output name
savename=URI[0:len(URI)-3] 

## set the output format to convert to
output_format = 'nc'

## we can either subset the data by a geometry from a shapefile, or convert to
## geojson for the entire spatial domain. there are other options here (i.e. a
## bounding box for tiling or a Shapely geometry).
GEOM = SHAPEFILE_PATH

mvars = 'discharge_mass_flux'
basins = ('CW', 'NW', 'NO', 'NE', 'SE', 'SW')
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
    ifile = ret
    print('path to output file: {0}'.format(ret))
    scalar_ofile = os.path.join(odir, prefix, '.'.join(['_'.join(['scalar_fldsum', prefix]), 'nc']))
    logger.info('Calculating field sum and saving to \n {}'.format(scalar_ofile))
    cdo.fldsum(input='-selvar,{} -seltimestep,2/10000 {}'.format(mvars, ifile), output=scalar_ofile)
    runmean_ofile = os.path.join(odir, prefix, '.'.join(['_'.join(['runmean_10yr', prefix]), 'nc']))
    logger.info('Calculating running mean and saving to \n {}'.format(runmean_ofile))
    cdo.runmean('10', input=scalar_ofile, output=runmean_ofile)
    anomaly_ofile = os.path.join(odir, prefix, '.'.join(['_'.join(['anomaly_runmean_10yr', prefix]), 'nc']))
    logger.info('Calculating anomalies and saving to \n {}'.format(anomaly_ofile))
    cdo.sub(input='{} -timmean -seltimestep,1/10 {}'.format(runmean_ofile, scalar_ofile), output=anomaly_ofile)
