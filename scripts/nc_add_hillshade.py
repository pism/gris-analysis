#!/usr/bin/env python
# Copyright (C) 2017 Andy Aschwanden

import os
import numpy as np
import logging
import logging.handlers
from argparse import ArgumentParser

from netCDF4 import Dataset as NC

# create logger
logger = logging.getLogger('hillshade')
logger.setLevel(logging.INFO)

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
# create formatter
formatter = logging.Formatter('%(name)s - %(message)s')

# add formatter to ch and fh
ch.setFormatter(formatter)

# add ch to logger
logger.addHandler(ch)

# See http://pubs.usgs.gov/of/1992/of92-422/of92-422.pdf
#     // W225 = sin^2(aspect - 225) = 0.5 * (1 - 2 * sin(aspect) * cos(aspect))
#     // W270 = sin^2(aspect - 270) = cos^2(aspect)
#     // W315 = sin^2(aspect - 315) = 0.5 * (1 + 2 * sin(aspect) * cos(aspect))
#     // W360 = sin^2(aspect - 360) = sin^2(aspect)
#     // hillshade=  0.5 * (W225 * hillshade(az=225) +
#     //                    W270 * hillshade(az=270) +
#     //                    W315 * hillshade(az=315) +
#     //                    W360 * hillshade(az=360))

class Hillshade(object):

    '''
    A class to add a hillshade to a netCDF time-series.

    Parameters
    ----------

    ifile: netCDF file with dimensions ('time', 'y', 'x'). Other permutations are currently not supported
    variable: variable used to create a hillshade

    kwargs
    ----------

    altitude:
    azimuth:
    fill_value:
    mask_variable: if variables_to_mask is not None, use this variable to mask them
    thk_threshold: if thickness_mask is True, use this threshold
    zf: 
    '''

    def __init__(self, ifile, variable='usurf', threshold_masking=True, variables_to_mask=None, multidirectional=False, *args, **kwargs):
        self.threshold_masking = threshold_masking
        self.do_masking = False
        self.ifile = ifile
        self.variable = variable
        if variables_to_mask is not None:
            self.variables_to_mask = variables_to_mask.split(',')
            self.do_masking = True
        else:
            self.variables_to_mask = variables_to_mask
        self.multidirectional = multidirectional
        self.params = {'altitude': 45,
                       'azimuth': 45,
                       'fill_value': 0,
                       'threshold_masking_variable': 'thk',
                       'threshold_masking_value': 10,
                       'zf': 1}
        for key in kwargs:
            if key in ('altitude', 'azimuth', 'fill_value', 'hillshade_var', 'zf'):
                self.params[key] = kwargs[key]

        filters = self._check_vars()
        self.dx = self._get_dx()
        self._create_vars(filters)

    def _check_vars(self):

        filters = ()
        logger.info('Checking for variables')
        nc = NC(self.ifile, 'r')
        for mvar in (['time'] + [self.variable]):
            if mvar in nc.variables:
                logger.info('variable {} found'.format(mvar))
                ncfilters = nc.variables[self.variable].filters()
            else:
                logger.info('variable {} NOT found'.format(mvar))

        if self.do_masking:
            for mvar in (self.variables_to_mask + [self.params['threshold_masking_variable']]):
                if mvar in nc.variables:
                    logger.info('variable {} found'.format(mvar))
                else:
                    logger.info('variable {} NOT found'.format(mvar))
        nc.close()
        
        return filters
    
           
    def _cart2pol(self, x, y):
        '''
        cartesian to polar coordinates
        '''
        theta = np.arctan2(y, x)
        rho = np.sqrt(x**2 + y**2)
        return (theta, rho) 

    def _create_vars(self, filters):
        '''
        create netCDF variables if they don't exist yet
        '''

        ifile = self.ifile
        nc = NC(ifile, 'a')
        variable = self.variable
        hs_var = self.variable + '_hs'
        if hs_var  not in nc.variables:
            hs = nc.createVariable(hs_var, 'i', dimensions=('time', 'y', 'x'), fill_value=self.params['fill_value'], *filters)
            hs.grid_mapping = 'mapping'
        nc.close()
                
    def _get_dx(self):
        
        nc = NC(self.ifile, 'r')

        x0, x1 = nc.variables['x'][0:2]
        y0, y1 = nc.variables['y'][0:2]
        
        nc.close()

        dx = x1 - x0
        dy = y1 - y0

        assert dx == dy

        return dx

    def _hillshade(self, dem):
       '''
       shaded relief using the ESRI algorithm
       '''

       # lighting azimuth
       azimuth = self.params['azimuth']
       azimuth = 360.0 - azimuth + 90 # convert to mathematic unit
       if (azimuth>360) or (azimuth==360):
          azimuth = azimuth - 360
       azimuth = azimuth * (np.pi / 180)  # convert to radians

       # lighting altitude
       altitude = self.params['altitude']
       altitude = (90 - altitude) * (np.pi / 180)  # convert to zenith angle in radians

       # calc slope and aspect (radians)
       dx = self.dx
       fx, fy = np.gradient(dem, dx)  # uses simple, unweighted gradient of immediate
       [asp, grad] = self._cart2pol(fy, fx)  # convert to carthesian coordinates

       zf = self.params['zf']
       grad = np.arctan(zf * grad)  # steepest slope
       # convert asp
       asp[asp<np.pi] = asp[asp < np.pi]+(np.pi / 2)
       asp[asp<0] = asp[asp<0] + (2 * np.pi)

       ## hillshade calculation
       h = 255.0 * ((np.cos(altitude) * np.cos(grad))
                    + (np.sin(altitude) * np.sin(grad) * np.cos(azimuth - asp)))
       h[h<0] = 0  # set hillshade values to min of 0.

       return h

    def run(self):
        logger.info('Processing file {}'.format(ifile))
        fill_value = self.params['fill_value']
        logger.info('Processing variable {}'.format(self.variable))
        hs_var = self.variable  + '_hs'
        nc = NC(ifile, 'a')
        nt = len(nc.variables['time'][:])
        for t in range(nt):
            logger.info('Processing time {} of {}'.format(t, nt))
            dem = nc.variables[self.variable][t, Ellipsis]
            hs = self._hillshade(dem)
            hs[dem==0] = fill_value
            nc.variables[hs_var][t, Ellipsis] = hs
            if self.threshold_masking:
                m = nc.variables[self.params['threshold_masking_variable']][t, Ellipsis]
                hs[m <= self.params['threshold_masking_value']] = fill_value
            if self.do_masking:
                for mvar in self.variables_to_mask:
                    mt = nc.variables[self.params['threshold_masking_variable']][t, Ellipsis]
                    m = nc.variables[mvar][t, Ellipsis]
                    try:
                        m_fill_value = nc.variables[mvar]._FillValue
                    except:
                        m_fill_value = fill_value
                    m[mt < self.params['threshold_masking_value']] = m_fill_value
                    nc.variables[mvar][t, Ellipsis] = m
            
        nc.close()


if __name__ == "__main__":

    # set up the option parser
    parser = ArgumentParser()
    parser.description = "Postprocessing files."
    parser.add_argument("FILE", nargs=1,
                        help="file", default=None)
    parser.add_argument("-z", dest='zf', type=float,
                        help="ZFactor", default=2.5)
    parser.add_argument("-v", "--variables", dest='variables',
                        help="Variables to create hillshade, comma-separated list. Default='usurf,topg'", default='usurf,topg')

    options = parser.parse_args()
    ifile = options.FILE[0]
    variables = options.variables.split(',')
    zf = options.zf
    multidirectional = False

    for m_var in variables:
        hs = Hillshade(ifile, variable=m_var, variables_to_mask='velsurf_mag,usurf_hs,usurf,thk', multidirectional=multidirectional, zf=zf)
        hs.run() 

