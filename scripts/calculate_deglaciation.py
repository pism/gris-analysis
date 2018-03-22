#!/usr/bin/env python
# Copyright (C) 2017, 2018 Andy Aschwanden

import os
import gdal
import numpy as np
import logging
import logging.handlers
from argparse import ArgumentParser

from netCDF4 import Dataset as NC
from netcdftime import utime

def copy_dimensions(input_file, output_file):
    """Copy dimensions (time, x, y) and corresponding coordinate variables
    from input_file to output_file.

    """
    # Create dimensions
    for dname, the_dim in input_file.dimensions.iteritems():
        output_file.createDimension(dname, len(the_dim) if not the_dim.isunlimited() else None)

    # Copy coordinate variables
    for v_name in ['x', 'y', 'time']:
        var_in = input_file.variables[v_name]
        var_out = output_file.createVariable(v_name, var_in.datatype, var_in.dimensions)

        # Copy variable attributes
        var_out.setncatts({k: var_in.getncattr(k) for k in var_in.ncattrs()})

        var_out[:] = var_in[:]

def process_block(time, H, H_threshold, t_min, output):
    """Process a block containing several time records of ice thickness,
    modifying output in place. Assumes that if an element of output is
    greated than or equal to t_min, then this location is already
    processed.

    """
    n_rows, n_cols = output.shape

    for r in  range(n_rows):
        for c in range(n_cols):
            if output[r, c] < t_min: # assume that the computed value is always greater than t_min
                try:
                    idx = np.where(H[:, r, c] < H_threshold)[0][0]
                    output[r, c] = time[idx] / secpera
                except:
                    pass

def block_size(shape, limit):
    """Return the block size to use when processing a variable with the
    number of elements given by shape, assuming that we have limit
    bytes of RAM available.

    """
    variable_size = np.prod(shape) * 8 # assuming 8 bytes per element (i.e. double)

    n_blocks = variable_size / float(limit)

    return int(np.floor(shape[0] / n_blocks))

def calc_deglaciation_time(infile, outfile, output_variable_name, thickness_threshold,
                           memory_limit):
    '''Calculate year of deglaciation (e.g. when ice thickness drops
    below threshold 'thickness_threshold')

    '''
    nc_in = NC(infile, 'r')
    nc_out = NC(outfile, 'w')

    copy_dimensions(nc_in, nc_out)

    time = nc_out.variables['time'][:]
    t_length = len(time)
    t_min = time[0]

    if output_variable_name not in nc_out.variables:
        deglac_time = nc_out.createVariable(output_variable_name, 'f',
                                            dimensions=('y', 'x'), fill_value=0)
    else:
        deglac_time = nc_out.variables[output_variable_name]
    deglac_time.long_name = 'year of deglaciation'

    thk = nc_in.variables["thk"]

    # set to a value below the first time record
    result = np.zeros_like(deglac_time) + t_min
    result[thk[0] >= thickness_threshold] = t_min - 1.0

    k = 0
    N = block_size(thk.shape, memory_limit)
    while k + N < t_length + 1:
        print("Processing records from {} to {}...".format(k, k + N - 1))
        H = thk[k:k + N]
        process_block(time[k:k + N], H, thickness_threshold, t_min, result)
        k += N

    if k < t_length:
        print("Processing records from {} to {}...".format(k, t_length - 1))
        H = thk[k:]
        process_block(time[k:], H, thickness_threshold, t_min, result)

    deglac_time[:] = result

    nc_in.close()
    nc_out.close()

def create_logger():
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

    return logger

if __name__ == "__main__":

    logger = create_logger()

    parser = ArgumentParser()
    parser.description = "Postprocessing files."
    parser.add_argument("FILE", nargs=1,
                        help="File to process", default=None)
    parser.add_argument("-m", help="Memory limit, in Mb", default=1024, type=int)
    options = parser.parse_args()

    input_file   = options.FILE[0]
    memory_limit = options.m * 2**20 # convert to bytes

    thickness_threshold = 10    # meters
    secpera = 24 * 3600 * 365   # for the 365_day calendar only

    # Process experiments
    dir_nc = 'deglaciation_time_nc'
    dir_gtiff = 'deglaciation_time_tif'
    output_variable_name = 'deglac_year'

    idir =  os.path.split(input_file)[0].split('/')[0]

    for dir_processed in (dir_gtiff, dir_nc):
        if not os.path.isdir(os.path.join(idir, dir_processed)):
            os.mkdir(os.path.join(idir, dir_processed))

    logger.info('Processing file {}'.format(input_file))
    exp_basename =  os.path.split(input_file)[-1].split('.nc')[0]
    exp_nc_wd = os.path.join(idir, dir_nc, exp_basename + '.nc')

    calc_deglaciation_time(input_file, exp_nc_wd, output_variable_name, thickness_threshold,
                           memory_limit)

    m_exp_nc_wd = 'NETCDF:{}:{}'.format(exp_nc_wd, output_variable_name)
    m_exp_gtiff_wd = os.path.join(idir, dir_gtiff,
                                  output_variable_name + '_' + exp_basename + '.tif')

    logger.info('Converting variable {} to GTiff and save as {}'.format(output_variable_name,
                                                                        m_exp_gtiff_wd))

    gdal_gtiff_options = gdal.TranslateOptions(format='GTiff', outputSRS='EPSG:3413')
    gdal.Translate(m_exp_gtiff_wd, m_exp_nc_wd, options=gdal_gtiff_options)
