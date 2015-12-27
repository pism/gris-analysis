#!/usr/bin/env python
# Copyright (C) 2014-2015 Andy Aschwanden

import itertools
import os
import subprocess
import multiprocessing
import time
from argparse import ArgumentParser

# Set up the option parser
parser = ArgumentParser()
parser.description = "Test scripts to use multiprocessing to extract contours."
parser.add_argument("INFILE", nargs='*')
parser.add_argument("-g", "--grid", dest="GRID", type=int,
                    choices=[18000, 9000, 4500, 3600, 1800, 1500, 1200, 900, 600, 450],
                    help="Horizontal grid resolution", default=1500)
parser.add_argument("-N", '--n_procs', dest="N", type=int,
                    help='''Number of cores/processors.''', default=7)
parser.add_argument("-t", "--type", dest="TYPE",
                    choices=['ctrl', 'old_bed', 'ba01_bed', '970mW_hs', 'jak_1985', 'cresis'],
                    help="Output size type", default='ctrl')
parser.add_argument("--dataset_version", dest="VERSION",
                    choices=['1.1', '1.2', '2'],
                    help="Input data set version", default='2')

options = parser.parse_args()
num_simultaneous_processes = options.N
grid_resolution = options.GRID
TYPE = options.TYPE
VERSION = options.VERSION
MYFILE = options.INFILE

def process_pair(value):
    process_time=time.time()
    subprocess.check_call(value, stderr=subprocess.STDOUT, shell=True)
    runtime=time.time()-process_time
    print 'done: %f seconds %s\n' % (runtime, value)
    return runtime

obsfile = 'observed/processed/jakobshavn_sar_velocities_1500m_2008-2009.nc'
geotiff = '--geotiff_file observed/figures/MODISJakobshavn250m.tif'
process_list = []
for infile in MYFILE:
    path, filename = os.path.split(infile)
    d = '/'.join([path, 'speed_contours'])
    if not os.path.exists(d):
        os.makedirs(d)
    outfile = d + '/' + filename.split('.nc')[0] + '_speed_contours.shp'

    
    process_string = 'gdal_contour -a speed -fl 100 300 1000 3000  NETCDF:{infile}:velsurf_mag {outfile}'.format(outfile=outfile, infile=infile)
    process_list.append(process_string)


start_time = time.time()
pool = multiprocessing.Pool(num_simultaneous_processes)
results = []
r = pool.map_async(process_pair, process_list, callback=results.append)
r.wait() # Wait on the results
print results


print '%f seconds for %d processes - %f seconds/process' % (time.time()-start_time,len(process_list),(time.time()-start_time)/len(process_list))
