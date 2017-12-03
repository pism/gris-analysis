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


gcm_marker_dict = {'CanESM2': 'd',
                   'MIROC5': 'o',
                   'NorESM1': 's'}
rcp_col_dict = {'45': '#fdae6b',
                '85': '#e6550d'}

col_dict = {
    'CanESM2': {'45': '#9ecae1', '85': '#3182bd'},
    'MIROC5': {'45': '#a1d99b', '85': '#31a354'},
    'NorESM1': {'45': '#bcbddc', '85': '#756bb1'}}

fig, ax = plt.subplots()

rus = []
tass = []
p_ols = []
for k, ifile in enumerate(options.FILE):
    print ifile, k
    nc = NC(ifile, 'r')
    forcing = nc.forcing
    gcm, rcp = forcing.split('-')
    rcp = rcp.split('rcp')[-1]
    tas = np.squeeze(nc.variables['ST'][:]) 
    ru = np.squeeze(nc.variables['RU'][:])
    tas -= tas.min()
    ru /= ru.min()

    rus.append(ru)
    tass.append(tas)
    print gcm, rcp
    ax.scatter(tas, ru, s=2.5, facecolors='none', edgecolors=col_dict[gcm][rcp], label='{}-RCP{}'.format(gcm, rcp), lw=0.5)

    tasS = pa.Series(data=tas, index=tas)
    ruS = pa.Series(data=ru, index=tas)
    # Perform Ordinary Least Squares regression analysis
    m_ols = sm.OLS(ruS, sm.add_constant(tasS)).fit()
    p_ols.append(m_ols)
    print m_ols.summary()
    m_ols.rsquared
    bias, trend = m_ols.params
    # ax.plot(tas, bias + trend * tas, color=col_dict[gcm][rcp], lw=0.2)
    nc.close()
    
ru_cat = rus[0]
tas_cat = tass[0]
for k in range(len(rus)):
    if k>0:
        ru_cat = np.concatenate((ru_cat, rus[k]))
        tas_cat = np.concatenate((tas_cat, tass[k]))

tasS = pa.Series(data=tas_cat, index=tas_cat)
ruS = pa.Series(data=ru_cat, index=tas_cat)
# Perform Ordinary Least Squares regression analysis
p_ols = sm.OLS(ruS, sm.add_constant(tasS)).fit()
print p_ols.summary()
bias, trend = p_ols.params
print bias, trend
ax.plot(tas_cat, bias + trend * tas_cat, color='k', lw=0.5)
ax.text(0.25, 3, 'r$^2$={:1.2f}'.format(p_ols.rsquared))
ax.text(0.25, 3.5, 'y={:1.2f}x + {:1.2f}'.format(trend, bias))
ax.set_ylabel('Runoff normalized (1)')
ax.set_xlim(0, 10)
legend = ax.legend(loc="upper right",
                   edgecolor='0',
                   bbox_to_anchor=(0, 0, 1., 1),
                   bbox_transform=plt.gcf().transFigure)
legend.get_frame().set_linewidth(0.2)

plt.savefig('tas_ru.pdf', bbox_inches='tight')

a = 0.5
C = 0.15
def m(x, alpha, beta, C, a):
    return (1 + C * (a * x)**alpha * x**beta)

T = np.linspace(0., 10, 101)
fig, ax = plt.subplots()
for alpha, beta in ([0.5, 1.0], [0.54, 1.17], [0.85, 1.61]):
    ax.plot(T, m(T, alpha, beta, C, a), label='$\\alpha$={}, $\\beta$={}'.format(alpha, beta))
ax.set_xlabel(u'Temperature anomaly (\u00B0C)')
ax.set_ylabel('Melt scale factor (1)')
ax.set_xlim(0, 10)
ax.set_ylim(0, 25)
ax.set_yticks([1, 5, 10, 15, 20, 25])
legend = ax.legend(loc="lower left",
                   edgecolor='0',
                   bbox_to_anchor=(0.15, 0.4, 0.2, 0.2),
                   bbox_transform=plt.gcf().transFigure)
legend.get_frame().set_linewidth(0.)

plt.savefig('temp2melt.pdf', bbox_inches='tight')
