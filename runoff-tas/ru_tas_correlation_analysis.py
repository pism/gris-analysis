#!/usr/bin/env python
# Copyright (C) 2017 Andy Aschwanden

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from netCDF4 import Dataset as NC
import numpy as np
import pylab as plt

import pandas as pa
import statsmodels.api as sm


parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.description = "Generating scripts for warming experiments."
parser.add_argument("FILE", nargs='*',
                    help="Input file to restart from", default=None)

options = parser.parse_args()

fontsize = 6
lw = 0.5
aspect_ratio = 0.35
markersize = 2
fig_width = 3.15  # inch
fig_height = aspect_ratio * fig_width  # inch
fig_size = [fig_width, fig_height]

params = {'backend': 'ps',
          'axes.linewidth': 0.25,
          'lines.linewidth': lw,
          'axes.labelsize': fontsize,
          'font.size': fontsize,
          'xtick.direction': 'in',
          'xtick.labelsize': fontsize,
          'xtick.major.size': 2.5,
          'xtick.major.width': 0.25,
          'ytick.direction': 'in',
          'ytick.labelsize': fontsize,
          'ytick.major.size': 2.5,
          'ytick.major.width': 0.25,
          'legend.fontsize': fontsize,
          'lines.markersize': markersize,
          'font.size': fontsize,
          'figure.figsize': fig_size}

plt.rcParams.update(params)

colors = ['#542788',
          '#b35806',
          '#e08214',
          '#fdb863',
          '#b2abd2',
          '#8073ac',
          '#000000']

fig, ax = plt.subplots()

for k, ifile in enumerate(options.FILE):
    print ifile, k
    nc = NC(ifile, 'r')

    tas = np.squeeze(nc.variables['ST'][:]) 
    ru = np.squeeze(nc.variables['RU'][:])
    tas -= tas.min()
    ru /= ru.min()
    ax.scatter(tas, ru, s=1, c=colors[k])

    tasS = pa.Series(data=tas, index=tas)
    ruS = pa.Series(data=ru, index=tas)
    # Perform Ordinary Least Squares regression analysis
    p_ols = sm.OLS(ruS, sm.add_constant(tasS)).fit()
    print p_ols.summary()
    nc.close()
    

    
plt.savefig('tas_ru.pdf')
plt.show()
