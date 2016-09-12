#!/usr/bin/env python
# Copyright (C) 2016 Andy Aschwanden


from argparse import ArgumentParser
import tempfile
import ocgis
import os


# set up the option parser
parser = ArgumentParser()
parser.description = "Generating scripts for prognostic simulations."
parser.add_argument("FILE", nargs=1)
parser.add_argument("--shape_file", dest="shape_file",
                    help="Path to shape file with basins", default=None)
parser.add_argument("--o_dir", dest="odir",
                    help="output directory", default='test')

options = parser.parse_args()
filename = options.FILE[0]
SHAPEFILE_PATH = options.shape_file

odir = options.odir
if not os.path.isdir(odir):
    os.mkdir(odir)

ocgis.env.OVERWRITE = True

## path to target file (may also be an OPeNDAP target)
URI = filename

## the target variable in the dataset to convert
VARIABLE = None
VARIABLE = 'discharge_flux'

# Create filename
savename=filename[0:len(filename)-3] 

## set the output format to convert to
output_format = 'nc'


## we can either subset the data by a geometry from a shapefile, or convert to
## geojson for the entire spatial domain. there are other options here (i.e. a
## bounding box for tiling or a Shapely geometry).
GEOM = SHAPEFILE_PATH
#GEOM = None

basins = range(1, 9)

## connect to the dataset and load the data as a field object. this will be used
## to iterate over time coordinates during the conversion step.
# rd = ocgis.RequestDataset(uri=URI, variable=VARIABLE)
# field = rd.get()

## selecting specific geometries from a shapefile requires knowing the target
## geometry's UGID inside the shapefile. shapefile are required to have this
## identifier at this time as a full attribute search is not enabled. this code
## searches for TX to find the UGID associated with that state.
    


## this again is the target dataset with a time range for subsetting now
#rd = ocgis.RequestDataset(uri=URI,variable=VARIABLE,time_range=[centroid,centroid])
rd = ocgis.RequestDataset(uri=URI, variable=VARIABLE)
for basin in basins:
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
                              snippet=True,
                              select_ugid=select_ugid,
                              output_format=output_format,
                              prefix=prefix,
                              dir_output=odir)
    ret = ops.execute()
    print('path to output file: {0}'.format(ret))

