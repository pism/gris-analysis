#!/usr/bin/env python
# Copyright (C) 2017 Andy Aschwanden

import os
import numpy as np
import logging
import logging.handlers
from argparse import ArgumentParser

from netCDF4 import Dataset as NC

# create logger
logger = logging.getLogger('postprocess')
logger.setLevel(logging.INFO)

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
# create formatter
formatter = logging.Formatter('%(name)s - %(module)s - %(message)s')

# add formatter to ch and fh
ch.setFormatter(formatter)

# add ch to logger
logger.addHandler(ch)


class Hillshade(object):

    '''
    A class to add a hillshade to a netCDF time-series.

    Parameters
    ----------

    '''

    def __init__(self, ifile, *args, **kwargs):
        super(Hillshade, self).__init__(*args, **kwargs)
        self.ifile = ifile
        self.params = {'altitude': 45,
                       'azimuth': 45,
                       'fill_value': 0,
                       'zf': 5}
        for key, value in kwargs:
            if key in ('altitude', 'azimuth', 'fill_value', 'hillshade_var', 'zf'):
                self.params[key] = value

        self._check_dims_and_vars()

    def _check_dims_and_vars(self):
       
        nc = NC(self.ifile, 'r')
        for mvar in ('time', 'thk'):
            if mvar in nc.variables:
                logger.info('variable {} found'.format(mvar))
            else:
                logger.info('variable {} NOT found'.format(mvar))
        #ds = nc.variables['run_stats'][]
        self.dx = 1500
        nc.close()
           
    def _cart2pol(self, x, y):
        '''
        cartesian to polar coordinates
        '''
        theta = np.arctan2(y, x)
        rho = np.sqrt(x**2 + y**2)
        return (theta, rho) 

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
        nc = NC(ifile, 'a')
        nt = len(nc.variables['time'][:])
        if 'usurf_hs' not in nc.variables:
            nc.createVariable('usurf_hs', 'i', dimensions=('time', 'y', 'x'), fill_value=fill_value)
        for t in range(nt):
            logger.info('Processing time {} of {}'.format(t, nt))
            dem = nc.variables['usurf'][t, Ellipsis]
            mthk = nc.variables['thk'][t, Ellipsis]
            mspeed = nc.variables['velsurf_mag'][t, Ellipsis]
            hs = self._hillshade(dem)
            hs[dem==0] = fill_value
            hs[mthk<=10] = fill_value
            mspeed[mthk<=10] = -2e9
            nc.variables['usurf_hs'][t, Ellipsis] = hs
            nc.variables['velsurf_mag'][t, Ellipsis] = mspeed
            
        nc.close()


if __name__ == "__main__":

    # set up the option parser
    parser = ArgumentParser()
    parser.description = "Postprocessing files."
    parser.add_argument("FILE", nargs=1,
                        help="file", default=None)

    options = parser.parse_args()
    ifile = options.FILE[0]

    hs = Hillshade(ifile)
    hs.run()
                            

