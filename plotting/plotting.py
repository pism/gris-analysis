#!/usr/bin/env python

# Copyright (C) 2017-18 Andy Aschwanden

from argparse import ArgumentParser
import re
import matplotlib.transforms as transforms
from matplotlib.ticker import FormatStrFormatter
from netCDF4 import Dataset as NC

import matplotlib as mpl
import matplotlib.cm as cmx
import matplotlib.colors as colors

from cdo import Cdo
cdo = Cdo()

import cf_units
import numpy as np
import pylab as plt

from unidecode import unidecode

try:
    from pypismtools import unit_converter, smooth
except:
    from pypismtools.pypismtools import unit_converter, smooth


def set_size(w,h, ax=None):
    """ w, h: width, height in inches """
    
    if not ax: ax=plt.gca()
    l = ax.figure.subplotpars.left
    r = ax.figure.subplotpars.right
    t = ax.figure.subplotpars.top
    b = ax.figure.subplotpars.bottom
    figw = float(w)/(r-l)
    figh = float(h)/(t-b)
    ax.figure.set_size_inches(figw, figh)

def gmtColormap(fileName, log_color=False, reverse=False):
    '''
    Import a CPT colormap from GMT.

    Parameters
    ----------
    fileName : a cpt file.

    Example
    -------
    >>> cdict = gmtColormap("mycolormap.cpt")
    >>> gmt_colormap = colors.LinearSegmentedColormap("my_colormap", cdict)

    Notes
    -----
    This code snipplet modified after
    http://www.mail-archive.com/matplotlib-users@lists.sourceforge.net/msg09547.html
    '''
    import colorsys
    import os

    try:
        try:
            f = open(fileName)
        except:
            # Check if it's a colormap provided in colormaps/
            basedir, fname = os.path.split(__file__)
            my_file = os.path.join(basedir, 'colormaps', fileName)
            f = open(my_file)
    except:
        print("file ", fileName, "not found")
        return None

    lines = f.readlines()
    f.close()

    x = []
    r = []
    g = []
    b = []
    colorModel = "RGB"
    for l in lines:
        ls = l.split()
        if l[0] == "#":
            if ls[-1] == "HSV":
                colorModel = "HSV"
                continue
            else:
                continue
        if ls[0] == "B" or ls[0] == "F" or ls[0] == "N":
            pass
        else:
            x.append(float(ls[0]))
            r.append(float(ls[1]))
            g.append(float(ls[2]))
            b.append(float(ls[3]))
            xtemp = float(ls[4])
            rtemp = float(ls[5])
            gtemp = float(ls[6])
            btemp = float(ls[7])

    x.append(xtemp)
    r.append(rtemp)
    g.append(gtemp)
    b.append(btemp)

    if reverse:
        r.reverse()
        g.reverse()
        b.reverse()

    x = np.array(x, np.float32)
    r = np.array(r, np.float32)
    g = np.array(g, np.float32)
    b = np.array(b, np.float32)
    if colorModel == "HSV":
        for i in range(r.shape[0]):
            rr, gg, bb = colorsys.hsv_to_rgb(r[i] / 360., g[i], b[i])
            r[i] = rr
            g[i] = gg
            b[i] = bb
    if colorModel == "HSV":
        for i in range(r.shape[0]):
            rr, gg, bb = colorsys.hsv_to_rgb(r[i] / 360., g[i], b[i])
            r[i] = rr
            g[i] = gg
            b[i] = bb
    if colorModel == "RGB":
        r = r / 255.
        g = g / 255.
        b = b / 255.

    if log_color:
        xNorm = np.zeros((len(x), ))
        xNorm[1::] = np.logspace(-1, 0, len(x) - 1)
        xNorm[1::-2] /= 4
    else:
        xNorm = (x - x[0]) / (x[-1] - x[0])

    red = []
    blue = []
    green = []
    for i in range(len(x)):
        red.append([xNorm[i], r[i], r[i]])
        green.append([xNorm[i], g[i], g[i]])
        blue.append([xNorm[i], b[i], b[i]])
    colorDict = {'red': red, 'green': green, 'blue': blue}
    return (colorDict)

    
    
basin_list = ['CW', 'NE', 'NO', 'NW', 'SE', 'SW']
rcp_list = ['26', '45', '85']

# Set up the option parser
parser = ArgumentParser()
parser.description = "A script for PISM output files to time series plots using pylab/matplotlib."
parser.add_argument("FILE", nargs='*')
parser.add_argument("--bounds", dest="bounds", nargs=2, type=float,
                    help="lower and upper bound for ordinate, eg. -1 1", default=None)
parser.add_argument("--time_bounds", dest="time_bounds", nargs=2, type=float,
                    help="lower and upper bound for abscissa, eg. 1990 2000", default=[2008, 3008])
parser.add_argument("-b", "--basin", dest="basin",
                    choices=basin_list,
                    help="Basin to plot", default='GRIS')
parser.add_argument("-c", "--colormap", dest="my_colormap",
                    help="Colormap for flowline plots", default='jet')
parser.add_argument("-l", "--labels", dest="labels",
                    help="comma-separated list with labels, put in quotes like 'label 1,label 2'", default=None)
parser.add_argument("-f", "--output_format", dest="out_formats",
                    help="Comma-separated list with output graphics suffix, default = pdf", default='pdf')
parser.add_argument("-n", "--parallel_threads", dest="openmp_n",
                    help="Number of OpenMP threads for operators such as enssstat, Default=1", default=1)
parser.add_argument("--no_legends", dest="do_legend", action="store_false",
                    help="Do not plot legend",
                    default=True)
parser.add_argument("-o", "--output_file", dest="outfile",
                    help="output file name without suffix, i.e. ts_control -> ts_control_variable", default='unnamed')
parser.add_argument("--step", dest="step", type=int,
                    help="step for plotting values, if time-series is very long", default=1)
parser.add_argument("--start_year", dest="start_year", type=float,
                    help='''Start year''', default=2008)
parser.add_argument("--rcp", dest="mrcp",
                    choices=rcp_list,
                    help="Which RCP", default=26)
parser.add_argument("--rotate_xticks", dest="rotate_xticks", action="store_true",
                    help="rotate x-ticks by 30 degrees, Default=False",
                    default=False)
parser.add_argument("-r", "--output_resolution", dest="out_res",
                    help='''Resolution ofoutput graphics in dots per
                  inch (DPI), default = 300''', default=300)
parser.add_argument("--runmean", dest="runmean", type=int,
                    help='''Calculate running mean''', default=None)
parser.add_argument("--plot", dest="plot",
                    help='''What to plot.''',
                    choices=['basin_mass',
                             'basin_d',
                             'basin_flux_partitioning',
                             'basin_cumulative_partitioning',
                             'cmip5',
                             'cmip5_rcp',
                             'ctrl_mass',
                             'ctrl_mass_anim',
                             'percent_mass',
                             'flux_partitioning',
                             'grid_pc',
                             'grid_res',
                             'per_basin_flux',
                             'per_basin_d',
                             'profile_combined',
                             'rcp_mass',
                             'rcp_accum',
                             'rcp_runoff',
                             'rcp_d',
                             'rcp_flux',
                             'station_usurf'],
                    default='rcp_mass')
parser.add_argument("--title", dest="title",
                    help='''Plot title.''', default=None)
parser.add_argument("--ctrl_file", dest="ctrl_file", nargs='*',
                    help='''Filename of ctrl run''', default=None)

options = parser.parse_args()
basin = options.basin
mrcp = options.mrcp
my_colormap = options.my_colormap
ifiles = options.FILE
if options.labels != None:
    labels = options.labels.split(',')
else:
    labels = None
bounds = options.bounds
do_legend = options.do_legend
runmean = options.runmean
time_bounds = options.time_bounds
openmp_n = options.openmp_n
out_res = options.out_res
outfile = options.outfile
out_formats = options.out_formats.split(',')
plot = options.plot
rotate_xticks = options.rotate_xticks
step = options.step
title = options.title
ctrl_file = options.ctrl_file
dashes = ['-', '--', '-.', ':', '-', '--', '-.', ':']

if openmp_n > 1:
    pthreads = '-P {}'.format(openmp_n)
else:
    pthreads = ''

dx, dy = 4. / out_res, -4. / out_res

# Conversion between giga tons (Gt) and millimeter sea-level equivalent (mmSLE)
gt2mmSLE = 1. / 365
gt2cmSLE = 1. / 365 / 10.
gt2mSLE = 1. / 365 / 1000.

start_year = options.start_year

# Plotting styles
shadow_color = '0.25'
numpoints = 1

fontsize = 6
lw = 0.65 
aspect_ratio = 0.35
markersize = 2
fig_width = 3.1  # inch
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


basin_col_dict = {'SW': '#542788',
                  'CW': '#b35806',
                  'NE': '#e08214',
                  'NO': '#fdb863',
                  'NW': '#b2abd2',
                  'SE': '#8073ac',
                  'GR': '#000000'}


rcp_col_dict = {'CTRL': 'k',
                '85': '#8c2d04',
                '45': '#f16913',
                '26': '#fdae6b'}

rcp_col_dict = {'CTRL': 'k',
                '85': '#990002',
                '45': '#5492CD',
                '26': '#003466'}

rcp_dict = {'26': 'RCP 2.6',
            '45': 'RCP 4.5',
            '85': 'RCP 8.5',
            'CTRL': 'CTRL'}

line_colors = ['#5492CD', '#C47900', '#004F00', '#003466', '#808080']

res_col_dict = {'450': '#006d2c',
                '600': '#31a354',
                '900': '#74c476',
                '1800': '#bae4b3',
                '3600': '#fcae91',
                '4500': '#fb6a4a',
                '9000': '#de2d26',
                '18000': '#a50f15'}

flux_to_mass_vars_dict = {'tendency_of_ice_mass_glacierized': 'ice_mass',
             'tendency_of_ice_mass_due_to_flow': 'flow_cumulative',
             'tendency_of_ice_mass_due_to_conservation_error': 'conservation_error_cumulative',
             'tendency_of_ice_mass_due_to_basal_mass_flux': 'basal_mass_flux_cumulative',
             'tendency_of_ice_mass_due_to_surface_mass_flux': 'surface_mass_flux_cumulative',
             'tendency_of_ice_mass_due_to_discharge': 'discharge_cumulative'}
flux_vars = list(flux_to_mass_vars_dict.keys())

flux_abbr_dict = {'tendency_of_ice_mass_glacierized': '$\dot \mathregular{M}$',
                  'tendency_of_ice_mass': '$\dot \mathregular{M}$',
                  'tendency_of_ice_mass_due_to_flow': 'divQ',
                  'tendency_of_ice_mass_due_to_conservation_error': '\dot e',
                  'tendency_of_ice_mass_due_to_basal_mass_flux': 'BMB',
                  'tendency_of_ice_mass_due_to_surface_mass_flux': 'SMB',
                  'tendency_of_ice_mass_due_to_discharge': 'D',
                  'surface_accumulation_rate': 'SN',
                  'surface_runoff_rate': 'RU',
                  'discharge_flux': 'D'}

flux_short_dict = {'tendency_of_ice_mass_glacierized': 'dmdt',
                  'tendency_of_ice_mass_due_to_flow': 'divq',
                  'tendency_of_ice_mass_due_to_conservation_error': 'e',
                  'tendency_of_ice_mass_due_to_basal_mass_flux': 'bmb',
                  'tendency_of_ice_mass_due_to_surface_mass_flux': 'smb',
                  'tendency_of_ice_mass_due_to_discharge': 'd'}


flux_style_dict = {'tendency_of_ice_mass': '-',
                   'tendency_of_ice_mass_due_to_flow': ':',
                   'tendency_of_ice_mass_due_to_conservation_error': ':',
                   'tendency_of_ice_mass_due_to_basal_mass_flux': '-.',
                   'tendency_of_ice_mass_due_to_surface_mass_flux': ':',
                   'tendency_of_ice_mass_due_to_discharge': '--',
                   'discharge': '--',
                   'discharge_flux': '--',
                   'surface_accumulation_rate': ':',
                   'surface_runoff_rate': '-.'}

mass_abbr_dict = {'ice_mass': 'M',
             'flow_cumulative': 'Q',
             'conservation_error_cumulative': 'e',
             'basal_mass_flux_cumulative': 'BMB',
             'surface_mass_flux_cumulative': 'SMB',
             'discharge_cumulative': 'D'}

mass_style_dict = {'ice_mass': '-',
             'flow_cumulative': ':',
             'conservation_error_cumulative': ':',
             'basal_mass_flux_cumulative': '-.',
             'surface_mass_flux_cumulative': ':',
             'discharge_cumulative': '--'}

flux_plot_vars = ['surface_accumulation_rate', 'surface_runoff_rate', 'surface_melt_rate', 'tendency_of_ice_mass_due_to_discharge', 'tendency_of_ice_mass_due_to_surface_mass_balance']
mass_plot_vars = ['ice_mass']

area_ounits = 'm2'
flux_ounits = 'Gt year-1'
specific_flux_ounits = 'kg m-2 year-1'
mass_ounits = 'Gt'

runmean_window = 11

lhs_params_dict = {'FICE': {'param_name': 'surface.pdd.factor_ice', 'vmin': 4, 'vmax': 12, 'scale_factor': 910, 'symb': '$f_{\mathregular{i}}$'},
                   'FSNOW': {'param_name': 'surface.pdd.factor_snow', 'vmin': 2, 'vmax': 6, 'scale_factor': 910, 'symb': '$f_{\mathregular{s}}$'},
                   'PRS': {'param_name': 'atmosphere.precip_exponential_factor_for_temperature', 'vmin': 5, 'vmax': 7, 'scale_factor': 100, 'symb': '$\omega$'},
                   'RFR': {'param_name': 'surface.pdd.refreeze', 'vmin': 25, 'vmax': 75, 'scale_factor': 100, 'symb': '$\psi$'}
}

def add_inner_title(ax, title, loc, size=None, **kwargs):
    '''
    Adds an inner title to a given axis, with location loc.

    from http://matplotlib.sourceforge.net/examples/axes_grid/demo_axes_grid2.html
    '''
    from matplotlib.offsetbox import AnchoredText
    from matplotlib.patheffects import withStroke
    if size is None:
        size = dict(size=plt.rcParams['legend.fontsize'])
    at = AnchoredText(title, loc=loc, prop=size,
                      pad=0., borderpad=0.5,
                      frameon=False, **kwargs)
    ax.add_artist(at)
    return at


def plot_cmip5_forcing(plot_var='delta_T'):

    fig = plt.figure()
    offset = transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)
    ax = fig.add_subplot(111)


    for k, rcp in reversed(list(enumerate(rcp_list[:]))):

        rcp_files = [f for f in ifiles if 'rcp{}'.format(rcp) in f]

        ensmin_file = [f for f in rcp_files if 'ENSMIN' in f][0]
        ensmax_file = [f for f in rcp_files if 'ENSMAX' in f][0]
        ensmean_file = [f for f in rcp_files if 'ENSMEAN' in f][0]
        
        ensmin_cdf = cdo.readCdf(ensmin_file)
        ensmax_cdf = cdo.readCdf(ensmax_file)
        ensmean_cdf = cdo.readCdf(ensmean_file)
        
        t = ensmin_cdf.variables['time'][:]
        ensmin_date = np.arange(start_year + step,
                         start_year + (len(t[:]) + 1) * step,
                         step) 

        t = ensmax_cdf.variables['time'][:]
        ensmax_date = np.arange(start_year + step,
                         start_year + (len(t[:]) + 1) * step,
                         step) 

        t = ensmean_cdf.variables['time'][:]
        ensmean_date = np.arange(start_year + step,
                         start_year + (len(t[:]) + 1) * step,
                         step) 

        ensmin_vals = np.squeeze(ensmin_cdf.variables[plot_var][:])
        ensmax_vals = np.squeeze(ensmax_cdf.variables[plot_var][:])
        ensmean_vals = np.squeeze(ensmean_cdf.variables[plot_var][:])

        ax.fill_between(ensmean_date, ensmin_vals, ensmax_vals,
                        alpha=0.25,
                        linewidth=0.0,
                        color=rcp_col_dict[rcp])
        
        ax.plot(ensmin_date, ensmin_vals, color=rcp_col_dict[rcp],
                linewidth=0.2)
        ax.plot(ensmax_date, ensmax_vals, color=rcp_col_dict[rcp],
                linewidth=0.2)
        ax.plot(ensmean_date, ensmean_vals, color=rcp_col_dict[rcp],
                label=rcp_dict[rcp], linewidth=lw)
    
    if do_legend:
        legend = ax.legend(
                           loc="upper right",
                           edgecolor='0',
                           bbox_to_anchor=(.2, 0, .7, 0.89),
                           bbox_transform=plt.gcf().transFigure)
        legend.get_frame().set_linewidth(0.0)
        legend.get_frame().set_alpha(0.0)
    
    ax.set_xlabel('Year')
    ax.set_ylabel('T-anomaly (K)')
        
    if time_bounds:
        ax.set_xlim(time_bounds[0], time_bounds[1])

    if bounds:
        ax.set_ylim(bounds[0], bounds[1])

    ymin, ymax = ax.get_ylim()

    ax.yaxis.set_major_formatter(FormatStrFormatter('%1.0f'))

    if rotate_xticks:
        ticklabels = ax.get_xticklabels()
        for tick in ticklabels:
                tick.set_rotation(30)
    else:
        ticklabels = ax.get_xticklabels()
        for tick in ticklabels:
            tick.set_rotation(0)
                    
    if title is not None:
            plt.title(title)

    set_size(2.44, 0.86)
        
    for out_format in out_formats:
        out_file = outfile + '_cmip5_forcing_'  + plot_var + '.' + out_format
        print("  - writing image %s ..." % out_file)
        fig.savefig(out_file, bbox_inches='tight', dpi=out_res)


def plot_cmip5_rcp(plot_var='delta_T'):


    fig, ax = plt.subplots(3, 1, sharex='col', figsize=[6, 4])
    fig.subplots_adjust(hspace=0.25, wspace=0.05)

    for k, rcp in enumerate(rcp_list[:]):

        rcp_files = [f for f in ifiles if 'rcp{}'.format(rcp) in f]

        gcm_files = [f for f in rcp_files if ('r1i1p1' in f) and not ('ENS' in f)]
        ensstdm1_file = [f for f in rcp_files if 'ensstdm1' in f][0]
        ensstdp1_file = [f for f in rcp_files if 'ensstdp1' in f][0]
        ensstdm1_cdf = cdo.readCdf(ensstdm1_file)
        ensstdp1_cdf = cdo.readCdf(ensstdp1_file)
        t = ensstdp1_cdf.variables['time'][:]
        cmip5_date = np.arange(start_year + step,
                         start_year + (len(t[:]) + 1) * step,
                         step) 

        ensstdm1_vals = np.squeeze(ensstdm1_cdf.variables[plot_var][:])
        ensstdp1_vals = np.squeeze(ensstdp1_cdf.variables[plot_var][:])
        ax[k].fill_between(cmip5_date,  ensstdm1_vals, ensstdp1_vals,
                           alpha=0.20,
                           linewidth=0.50,
                           color='k')

        for m, gcm_file in enumerate(gcm_files):
            gcm = gcm_file.split('tas_Amon_')[1].split('_rcp')[0]
            gcm_cdf = cdo.readCdf(gcm_file)
            t = gcm_cdf.variables['time'][:]
            gcm_date = np.arange(start_year + step,
                         start_year + (len(t[:]) + 1) * step,
                         step) 
            gcm_vals = np.squeeze(gcm_cdf.variables[plot_var][:])
            ax[k].plot(gcm_date, gcm_vals, label=gcm, color=line_colors[m],
                       linewidth=0.4)
            
        ensmean_file = [f for f in rcp_files if ('r1i1p1' in f) and ('ENSMEAN' in f)][0]
        ensmean_cdf = cdo.readCdf(ensmean_file)
        t = ensmean_cdf.variables['time'][:]
        ensmean_date = np.arange(start_year + step,
                                 start_year + (len(t[:]) + 1) * step,
                                 step) 
        ensmean_vals = np.squeeze(ensmean_cdf.variables[plot_var][:])
        gcm = ensmean_file.split('tas_Amon_')[1].split('_rcp')[0]
        ax[k].plot(ensmean_date, ensmean_vals, label=gcm, color='k',
                linewidth=lw)

        if time_bounds:
            ax[k].set_xlim(time_bounds[0], time_bounds[1])

        if bounds:
            ax[k].set_ylim(bounds[0], bounds[1])

        ymin, ymax = ax[k].get_ylim()
        ax[k].yaxis.set_major_formatter(FormatStrFormatter('%1.0f'))
        
        ax[k].set_title('{}'.format(rcp_dict[rcp]))

    ax[2].set_xlabel('Year')
    ax[1].set_ylabel('T-anomaly (K)')

    if do_legend:
        legend = ax[2].legend(loc="lower left",
                              edgecolor='0',
                              bbox_to_anchor=(.13, 0.16, 0, 0),
                              bbox_transform=plt.gcf().transFigure)
        legend.get_frame().set_linewidth(0.0)
        legend.get_frame().set_alpha(0.0)

    if rotate_xticks:
        ticklabels = ax[2].get_xticklabels()
        for tick in ticklabels:
                tick.set_rotation(30)
    else:
        ticklabels = ax[2].get_xticklabels()
        for tick in ticklabels:
            tick.set_rotation(0)

    if title is not None:
            plt.title(title)


    for out_format in out_formats:
        out_file = outfile + '_cmip5_'  + plot_var + '.' + out_format
        print("  - writing image %s ..." % out_file)
        fig.savefig(out_file, bbox_inches='tight', dpi=out_res)

        
def plot_profile_ts_combined():

    try:
        cmap = getattr(plt.cm, my_colormap)
    except:
        # import and convert colormap
        cdict = gmtColormap(my_colormap)
        cmap = mpl.colors.LinearSegmentedColormap('my_colormap', cdict, 1024)
    
    mcm = cm = plt.get_cmap(cmap)

    nc = NC(ifiles[0], 'r')
    profile_names = nc.variables['profile_name'][:]
    for k, profile in enumerate(profile_names):

        print((u'Processing {} profile'.format(profile)))
        
        fig, ax = plt.subplots(3, 1, sharex='col', figsize=[3, 2])
        fig.subplots_adjust(hspace=0.15, wspace=0.05)
        
        profile_iunits = nc.variables['profile'].units
        profile_ounits = 'km'
        profile_vals = nc.variables['profile'][k, :]
        profile_vals = unit_converter(profile_vals, profile_iunits, profile_ounits)

        t_var = nc.variables['time'][:]
        date = np.arange(start_year,
                         start_year + (len(t_var[:]) + 1),
                         step)
        ma = np.where(date == time_bounds[0])[0][0]
        me = np.where(date == time_bounds[1])[0][0]

        plot_times = np.arange(ma, me+1, step)

        cNorm = colors.Normalize(vmin=time_bounds[0], vmax=time_bounds[1])
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=mcm)
        speed_vals_0 = np.nanmean(nc.variables['velsurf_mag'][k, 0:20, :], axis=0)

        for t in plot_times:
            colorVal = scalarMap.to_rgba(date[t])
            speed_vals = nc.variables['velsurf_mag'][k, t, :] - speed_vals_0
            speed_vals = nc.variables['velsurf_mag'][k, t, :]
            # speed_basal_vals = nc.variables['velbase_mag'][k, t, :]
            mask = (speed_vals < 1)
            speed_vals = np.ma.array(speed_vals, mask = mask)
            # speed_basal_vals = np.ma.array(speed_basal_vals, mask = mask)
            # slip_ratio = speed_vals / speed_basal_vals
            topg_vals = nc.variables['topg'][k, t, :]
            thk_vals = nc.variables['thk'][k, t, :]
            usurf_vals = nc.variables['usurf'][k, t, :]
            thk_mask = (thk_vals <= 20)
            thk_vals = np.ma.array(thk_vals, mask=thk_mask)
            usurf_mask = (usurf_vals < 100)
            usurf_mask = np.logical_or((usurf_vals < topg_vals), thk_mask)
            usurf_vals = np.ma.array(usurf_vals, mask=usurf_mask)
            ax[0].plot(profile_vals, speed_vals * thk_vals / 1e6, color=colorVal, linewidth=0.3)
            ax[1].plot(profile_vals, speed_vals, color=colorVal, linewidth=0.3)
            try:
                ax[2].plot(profile_vals, usurf_vals, color=colorVal, linewidth=0.3)
                bottom_vals = np.maximum(usurf_vals - thk_vals, topg_vals)
                ax[2].plot(profile_vals, np.ma.array(bottom_vals, mask=thk_mask), color=colorVal, linewidth=0.3)
            except:
                pass
            if t == plot_times[-1]:
                ax[2].plot(profile_vals, topg_vals, color='k', linewidth=0.3)
                try:
                    ax[2].plot(profile_vals, usurf_vals, color='k', linewidth=0.1)
                    bottom_vals = np.maximum(usurf_vals - thk_vals, topg_vals)
                    ax[2].plot(profile_vals, np.ma.array(bottom_vals, mask=thk_mask), color='k', linewidth=0.1)
                except:
                    pass
                

        xmin, xmax = ax[1].get_xlim()
        ymin, ymax = ax[1].get_ylim()

        ymin = -1200
        
        ax[0].set_ylabel('Flux\n (km$^{\mathregular{2}}$ yr$^{\mathregular{-1}}$)', multialignment='center')
        ax[1].set_ylabel('Speed\n (m yr$^{\mathregular{-1}}$)', multialignment='center')
        ax[2].fill_between([xmin, xmax], [ymin, ymin], color='#c6dbef', linewidth=0)
        tz = ax[2].fill_between(profile_vals, topg_vals * 0 + ymin, topg_vals, color='#fdbe85', linewidth=0)
        ax[2].set_ylabel('Altitude\n (masl)', multialignment='center')
        tz = ax[2].axhline(profile_vals[0], linestyle='solid', color='k', linewidth=0.3)
        tz.set_zorder(-1)
        ax[2].set_xlabel('distance ({})'.format(profile_ounits))

        ax[2].set_xlim(np.nanmin(profile_vals), np.nanmax(profile_vals))
        ax[2].set_ylim(np.nanmin(topg_vals))
        ax[2].yaxis.set_major_formatter(FormatStrFormatter('%1.0f'))

        if bounds:
            ax[1].set_ylim(bounds[0], bounds[1])
            
        ax[0].set_xlim(0, 65)
        ax[1].set_xlim(0, 65)
        ax[2].set_xlim(0, 65)

        if rotate_xticks:
            ticklabels = ax[1].get_xticklabels()
            for tick in ticklabels:
                tick.set_rotation(30)
        else:
            ticklabels = ax[1].get_xticklabels()
            for tick in ticklabels:
                tick.set_rotation(0)
                    
        if title is not None:
            plt.title(title)

        for out_format in out_formats:
            out_file = outfile + '_{}'.format(unidecode(profile).replace(' ', '_')) + '.' + out_format
            print("  - writing image %s ..." % out_file)
            fig.savefig(out_file, bbox_inches='tight', dpi=out_res)

    nc.close()


def plot_point_ts(plot_var='usurf'):

    nc0 = NC(ifiles[0], 'r')
    station_names = nc0.variables['station_name'][:]
    nc0.close()
    for k, station in enumerate(station_names):
    
        fig = plt.figure()
        offset = transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)
        ax = fig.add_subplot(111)
        
        for m, rcp in enumerate(rcp_list):

            rcp_file = [f for f in ifiles if 'rcp_{}'.format(rcp) in f][0]
            nc = NC(rcp_file, 'r')
            t = nc.variables['time'][:]
            date = np.arange(start_year + step,
                             start_year + (len(t[:]) + 1) ,
                             step) 
            var_vals = nc.variables[plot_var][k, :]
            ax.plot(date, var_vals, color=rcp_col_dict[rcp], label=rcp_dict[rcp])
            rel_change = - (var_vals[:] - var_vals[0]) / var_vals[0] * 100
            for pc in [2]:
                try:
                    idx = np.where(rel_change >= pc)[0][0]
                    ax.axvline(date[idx],
                               linewidth=0.2,
                               linestyle='dashed',
                               color=rcp_col_dict[rcp])
                except:
                    pass
            nc.close()

        if do_legend:
            legend = ax.legend(loc="lower left",
                               edgecolor='0',
                               bbox_to_anchor=(.12, 0.1, 0, 0),
                               bbox_transform=plt.gcf().transFigure)
            legend.get_frame().set_linewidth(0.0)
            legend.get_frame().set_alpha(0.0)


            
        
        ax.set_xlabel('Year')
        ax.set_ylabel('elevation change (m)')
                
        if time_bounds:
            ax.set_xlim(time_bounds[0], time_bounds[1])

        if bounds:
            ax.set_ylim(bounds[0], bounds[1])

        ymin, ymax = ax.get_ylim()

        ax.yaxis.set_major_formatter(FormatStrFormatter('%1.0f'))

        if rotate_xticks:
            ticklabels = ax.get_xticklabels()
            for tick in ticklabels:
                tick.set_rotation(30)
        else:
            ticklabels = ax.get_xticklabels()
            for tick in ticklabels:
                tick.set_rotation(0)
                    
        if title is not None:
            plt.title(title)

        for out_format in out_formats:
            out_file = outfile + '_{}_'.format(station)  + plot_var + '.' + out_format
            print("  - writing image %s ..." % out_file)
            fig.savefig(out_file, bbox_inches='tight', dpi=out_res)
    

def plot_ctrl_mass(plot_var=mass_plot_vars):
    
    fig = plt.figure()
    offset = transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)
    ax = fig.add_subplot(111)

    sle = 7.21
    ax.axhline(sle,
               color='k',
               linestyle='dashed',
               linewidth=0.2)

    for k, rcp in enumerate(rcp_list[::-1]):
        rcp_file = [f for f in ifiles if 'rcp_{}'.format(rcp) in f][0]
        cdf = cdo.readCdf(rcp_file)
        t = cdf.variables['time'][:]
        date = np.arange(start_year + step,
                         start_year + (len(t[:]) + 1) ,
                         step) 
        var_vals = cdf.variables[plot_var][:] - cdf.variables[plot_var][0]
        iunits = cdf.variables[plot_var].units
        var_vals = -unit_converter(var_vals, iunits, mass_ounits) * gt2mSLE

        plt.plot(date[var_vals<sle], var_vals[var_vals<sle],
                 color=rcp_col_dict[rcp],
                 linewidth=lw,
                 label=rcp_dict[rcp],)
    
    if do_legend:
        legend = ax.legend(loc="center right",
                           edgecolor='0',
                           bbox_to_anchor=(0.91, .63),
                           bbox_transform=plt.gcf().transFigure)
        legend.get_frame().set_linewidth(0.0)
        legend.get_frame().set_alpha(0.0)

    ax.set_xlabel('Year')
    ax.set_ylabel('$\Delta$(GMSL) (m)')
        
    if time_bounds:
        ax.set_xlim(time_bounds[0], time_bounds[1])

    if bounds:
        ax.set_ylim(bounds[0], bounds[1])

    ymin, ymax = ax.get_ylim()

    ax.yaxis.set_major_formatter(FormatStrFormatter('%1.2f'))

    if rotate_xticks:
        ticklabels = ax.get_xticklabels()
        for tick in ticklabels:
                tick.set_rotation(30)
    else:
        ticklabels = ax.get_xticklabels()
        for tick in ticklabels:
            tick.set_rotation(0)
                    
    if title is not None:
            plt.title(title)

    for out_format in out_formats:
        out_file = outfile + '_ctrl' + '_'  + plot_var + '.' + out_format
        print("  - writing image %s ..." % out_file)
        fig.savefig(out_file, bbox_inches='tight', dpi=out_res)

def plot_ctrl_mass_anim(plot_var=mass_plot_vars):

    rcp_26_file = [f for f in ifiles if 'rcp_{}'.format(26) in f][0]
    cdf_26 = cdo.readCdf(rcp_26_file)
    t_26 = cdf_26.variables['time'][:]
    date_26 = np.arange(start_year + step,
                     start_year + (len(t_26[:]) + 1) ,
                     step) 
    var_vals_26 = cdf_26.variables[plot_var][:] - cdf_26.variables[plot_var][0]
    iunits = cdf_26.variables[plot_var].units
    var_vals_26 = -unit_converter(var_vals_26, iunits, mass_ounits) * gt2mSLE

    rcp_45_file = [f for f in ifiles if 'rcp_{}'.format(45) in f][0]
    cdf_45 = cdo.readCdf(rcp_45_file)
    t_45 = cdf_45.variables['time'][:]
    date_45 = np.arange(start_year + step,
                     start_year + (len(t_45[:]) + 1) ,
                     step) 
    var_vals_45 = cdf_45.variables[plot_var][:] - cdf_45.variables[plot_var][0]
    iunits = cdf_45.variables[plot_var].units
    var_vals_45 = -unit_converter(var_vals_45, iunits, mass_ounits) * gt2mSLE

    rcp_85_file = [f for f in ifiles if 'rcp_{}'.format(85) in f][0]
    cdf_85 = cdo.readCdf(rcp_85_file)
    t_85 = cdf_85.variables['time'][:]
    date_85 = np.arange(start_year + step,
                     start_year + (len(t_85[:]) + 1) ,
                     step) 
    var_vals_85 = cdf_85.variables[plot_var][:] - cdf_85.variables[plot_var][0]
    iunits = cdf_85.variables[plot_var].units
    var_vals_85 = -unit_converter(var_vals_85, iunits, mass_ounits) * gt2mSLE

    for frame in range(1000):

        fig = plt.figure()
        offset = transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)
        ax = fig.add_subplot(111)            

        plt.plot(date_26, var_vals_26,
                 color=rcp_col_dict['26'],
                 linewidth=lw,
                 alpha=0.5)

        plt.plot(date_26[:frame], var_vals_26[:frame],
                 color=rcp_col_dict['26'],
                 linewidth=lw,
                 label=rcp_dict['26'],)

        plt.plot(date_26[frame], var_vals_26[frame],
                 color=rcp_col_dict['26'],
                 marker='o',
                 markersize=3,
                 linewidth=0)

        plt.plot(date_45, var_vals_45,
                 color=rcp_col_dict['45'],
                 linewidth=lw,
                 alpha=0.5)

        plt.plot(date_45[:frame], var_vals_45[:frame],
                 color=rcp_col_dict['45'],
                 linewidth=lw,
                 label=rcp_dict['45'],)

        plt.plot(date_45[frame], var_vals_45[frame],
                 color=rcp_col_dict['45'],
                 marker='o',
                 markersize=3,
                 linewidth=0)

        plt.plot(date_85, var_vals_85,
                 color=rcp_col_dict['85'],
                 linewidth=lw,
                 alpha=0.5)

        plt.plot(date_85[:frame], var_vals_85[:frame],
                 color=rcp_col_dict['85'],
                 linewidth=lw,
                 label=rcp_dict['85'],)

        plt.plot(date_85[frame], var_vals_85[frame],
                 color=rcp_col_dict['85'],
                 marker='o',
                 markersize=3,
                 linewidth=0)

        legend = ax.legend(loc="center left",
                           edgecolor='0',
                           bbox_to_anchor=(0.15, .65),
                           bbox_transform=plt.gcf().transFigure)
        legend.get_frame().set_linewidth(0.0)
        legend.get_frame().set_alpha(0.0)

        ax.set_xlabel('Year')
        ax.set_ylabel('sea level contribution (m)')

        if time_bounds:
            ax.set_xlim(time_bounds[0], time_bounds[1])

        if bounds:
            ax.set_ylim(bounds[0], bounds[1])

        ymin, ymax = ax.get_ylim()

        ax.yaxis.set_major_formatter(FormatStrFormatter('%1.2f'))

        if rotate_xticks:
            ticklabels = ax.get_xticklabels()
            for tick in ticklabels:
                    tick.set_rotation(30)
        else:
            ticklabels = ax.get_xticklabels()
            for tick in ticklabels:
                tick.set_rotation(0)

        if title is not None:
                plt.title(title)

        for out_format in out_formats:
            out_file = 'dgmsl_'  + plot_var + '_{:04d}.{}'.format(frame, 'png') 
            print("  - writing image %s ..." % out_file)
            fig.savefig(out_file, bbox_inches='tight', dpi=300)

        plt.close(fig)

        
def plot_percent_mass(plot_var=mass_plot_vars):
    
    fig = plt.figure(figsize=[4, 1.5])
    offset = transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)
    ax = fig.add_subplot(111)

    for pc in [5, 10, 20, 40, 60]:
        ax.axhline(pc, color='k', linestyle='dotted')
    for k, rcp in enumerate(rcp_list[::-1]):
        rcp_file = [f for f in ifiles if 'rcp_{}'.format(rcp) in f][0]
        cdf = cdo.readCdf(rcp_file)
        t = cdf.variables['time'][:]
        date = np.arange(start_year + step,
                         start_year + (len(t[:]) + 1) ,
                         step) 
        var_vals = cdf.variables[plot_var][:]
        for pc in [5, 10, 20, 40, 60]:
            try:
                idx = np.where(var_vals>= pc)[0][0]
                m_year = date[idx]
            except:
                m_year = np.nan
            print(('RCP {}, {}% mass lost in Year {}'.format(rcp, pc, m_year)))

        try:
            m_year = 4992
            print(('RCP {}, {}% mass lost in Year {}'.format(rcp, var_vals[m_year], m_year)))
        except:
            pass

        ax.semilogy(date, var_vals,
                 color=rcp_col_dict[rcp],
                 linewidth=lw,
                 label=rcp_dict[rcp],)
    
    if do_legend:
        legend = ax.legend(loc="center right",
                           edgecolor='0',
                           bbox_to_anchor=(0.88, .26),
                           bbox_transform=plt.gcf().transFigure)
        legend.get_frame().set_linewidth(0.0)
        legend.get_frame().set_alpha(0.0)

    ax.set_ylabel('Mass loss (%)')
    ax.set_xlabel('Year')

    ax.set_yticks([2, 5, 10, 20, 40, 60, 100])
    
    if time_bounds:
        ax.set_xlim(time_bounds[0], time_bounds[1])

    if bounds:
        ax.set_ylim(bounds[0], bounds[1])

    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    ax.xaxis.set_major_formatter(FormatStrFormatter('%1.0f'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%1.0f'))

    ax_sle = ax.twinx()
    ax_sle.set_ylim(0.0723, 7.23)
    ax_sle.set_ylabel('$\Delta$(GMSL) (m)')
    ax_sle.set_yscale('log')
    ax_sle.yaxis.set_major_formatter(FormatStrFormatter('%1.1f'))
    ax_sle.set_yticks([0.1, 0.5, 1, 2, 7])

    rotate_xticks = False
    if rotate_xticks:
        ticklabels = ax.get_xticklabels()
        for tick in ticklabels:
                tick.set_rotation(30)
    else:
        ticklabels = ax.get_xticklabels()
        for tick in ticklabels:
            tick.set_rotation(0)
                    
    if title is not None:
            plt.title(title)

    for out_format in out_formats:
        out_file = outfile + '_percent' + '_'  + plot_var + '.' + out_format
        print("  - writing image %s ..." % out_file)
        fig.savefig(out_file, bbox_inches='tight', dpi=out_res)



def plot_grid_res(plot_var='tendency_of_ice_mass_due_to_discharge'):
    
    for k, rcp in enumerate(['45']):

        fig = plt.figure()
        offset = transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)
        ax = fig.add_subplot(111)

        print(('Reading RCP {} files'.format(rcp)))
        rcp_files = [f for f in ifiles if 'rcp_{}'.format(rcp) in f]

        for m_file in rcp_files:
            dr = re.search('gris_g(.+?)m', m_file).group(1)
            cdf = cdo.readCdf(m_file)
            
            t = cdf.variables['time'][:]

            vals = cdf.variables[plot_var][:]
            iunits = cdf[plot_var].units
            vals = unit_converter(vals, iunits, flux_ounits) 

            date = np.arange(start_year + step,
                             start_year + (len(t[:]) + 1) ,
                             step) 

            ax.plot(date[:], vals,
                    color=res_col_dict[dr],
                    alpha=0.5,
                    linewidth=0.3)

        for m_file in rcp_files:
            dr = re.search('gris_g(.+?)m', m_file).group(1)
            cdf = cdo.runmean('11', input=m_file, returnCdf=True, options=pthreads)
            
            t = cdf.variables['time'][:]

            vals = cdf.variables[plot_var][:]
            iunits = cdf[plot_var].units
            vals = unit_converter(vals, iunits, flux_ounits) 

            date = np.arange(start_year + step,
                             start_year + (len(t[:]) + 1) ,
                             step) 

            ax.plot(date[:], vals,
                    color=res_col_dict[dr],
                    linewidth=lw,
                    label=dr)
        
        ax.set_xlabel('Year')
        ax.set_ylabel('Rate (Gt yr$^{\mathregular{-1}}$)')
            
        if time_bounds:
            ax.set_xlim(time_bounds[0], time_bounds[1])

        if bounds:
            ax.set_ylim(bounds[0], bounds[1])

        ymin, ymax = ax.get_ylim()

        ax.yaxis.set_major_formatter(FormatStrFormatter('%1.0f'))

        if rotate_xticks:
            ticklabels = ax.get_xticklabels()
            for tick in ticklabels:
                tick.set_rotation(30)
        else:
            ticklabels = ax.get_xticklabels()
            for tick in ticklabels:
                tick.set_rotation(0)
                    
        if do_legend:
            legend = ax.legend(loc="center right",
                               edgecolor='0',
                               bbox_to_anchor=(1.1, 0.5),
                               bbox_transform=plt.gcf().transFigure)
            legend.get_frame().set_linewidth(0.0)
            legend.get_frame().set_alpha(0.0)
            
            # handles, labels = ax.get_legend_handles_labels()
            # labels = [int(f) for f in labels]
            # # sort both labels and handles by labels
            # labels, handles = zip(*sorted(zip(labels, handles), key=int))
            # ax.legend(handles, labels)


        if title is not None:
            plt.title(title)

        for out_format in out_formats:
            out_file = outfile + '_rcp_{}_grid'.format(rcp) + '_'  + plot_var + '.' + out_format
            print("  - writing image %s ..." % out_file)
            fig.savefig(out_file, bbox_inches='tight', dpi=out_res)
        
def plot_grid_pc(plot_var='limnsw'):

    for k, rcp in enumerate(rcp_list[::-1]):

        fig = plt.figure()
        offset = transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)
        ax = fig.add_subplot(111)

        print(('Reading RCP {} files'.format(rcp)))
        rcp_files = [f for f in ifiles if 'rcp_{}'.format(rcp) in f]

        for m_file in rcp_files:
            dr = re.search('gris_g(.+?)m', m_file).group(1)
            cdf = cdo.runmean('11', input=m_file, returnCdf=True, options=pthreads)
            
            t = cdf.variables['time'][:]

            vals = cdf.variables[plot_var][:]

            date = np.arange(start_year + step,
                             start_year + (len(t[:]) + 1) ,
                             step) 

            ax.plot(date[:], vals,
                    color=res_col_dict[dr],
                    linewidth=lw,
                    label=dr)

            for pc in [1, 2, 5, 10]:
                try:
                    idx = np.where(vals>= pc)[0][0]
                    m_year = date[idx]
                except:
                    m_year = np.nan
                print(('{}m: {}% mass lost in Year {}'.format(dr, pc, m_year)))            

        
        ax.set_xlabel('Year')
        ax.set_ylabel('Mass loss (%)')
            
        if time_bounds:
            ax.set_xlim(time_bounds[0], time_bounds[1])

        if bounds:
            ax.set_ylim(bounds[0], bounds[1])

        ymin, ymax = ax.get_ylim()

        ax.yaxis.set_major_formatter(FormatStrFormatter('%1.0f'))

        if rotate_xticks:
            ticklabels = ax.get_xticklabels()
            for tick in ticklabels:
                tick.set_rotation(30)
        else:
            ticklabels = ax.get_xticklabels()
            for tick in ticklabels:
                tick.set_rotation(0)
                    
        if do_legend:
            legend = ax.legend(loc="lower left",
                               edgecolor='0',
                               bbox_to_anchor=(0.12, 0.25, 0, 0),
                               bbox_transform=plt.gcf().transFigure)
            legend.get_frame().set_linewidth(0.0)
            legend.get_frame().set_alpha(0.0)
            
            # handles, labels = ax.get_legend_handles_labels()
            # labels = [int(f) for f in labels]
            # # sort both labels and handles by labels
            # labels, handles = zip(*sorted(zip(labels, handles), key=int))
            # ax.legend(handles, labels)


        if title is not None:
            plt.title(title)

        for out_format in out_formats:
            out_file = outfile + '_rcp_{}_grid_percent'.format(rcp) + '_'  + plot_var + '.' + out_format
            print("  - writing image %s ..." % out_file)
            fig.savefig(out_file, bbox_inches='tight', dpi=out_res)
        


def plot_rcp_mass(plot_var=mass_plot_vars):
    
    fig = plt.figure()
    offset = transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)
    ax = fig.add_subplot(111)
    for k, rcp in enumerate(rcp_list[::-1]):

        print(('Reading RCP {} files'.format(rcp)))
        rcp_files = [f for f in ifiles if 'rcp_{}'.format(rcp) in f]

        pctl16_file = [f for f in rcp_files if 'enspctl16' in f][0]
        pctl84_file = [f for f in rcp_files if 'enspctl84' in f][0]

        cdf_enspctl16 = cdo.readCdf(pctl16_file)
        cdf_enspctl84 = cdo.readCdf(pctl84_file)
        t = cdf_enspctl16.variables['time'][:]

        enspctl16 = cdf_enspctl16.variables[plot_var][:]
        enspctl16_vals = cdf_enspctl16.variables[plot_var][:] - cdf_enspctl16.variables[plot_var][0]
        iunits = cdf_enspctl16[plot_var].units
        enspctl16_vals = -unit_converter(enspctl16_vals, iunits, mass_ounits) * gt2mSLE

        enspctl84 = cdf_enspctl84.variables[plot_var][:]
        enspctl84_vals = cdf_enspctl84.variables[plot_var][:] - cdf_enspctl84.variables[plot_var][0]
        iunits = cdf_enspctl84[plot_var].units
        enspctl84_vals = -unit_converter(enspctl84_vals, iunits, mass_ounits) * gt2mSLE

        date = np.arange(start_year + step,
                         start_year + (len(t[:]) + 1) ,
                         step) 


        # ensemble between 16th and 84th quantile
        ax.fill_between(date[:], enspctl16_vals, enspctl84_vals,
                        color=rcp_col_dict[rcp],
                        alpha=0.4,
                        linewidth=0)


        ax.plot(date[:], enspctl16_vals,
                color=rcp_col_dict[rcp],
                linestyle='solid',
                linewidth=0.4)

        ax.plot(date[:], enspctl84_vals,
                color=rcp_col_dict[rcp],
                linestyle='solid',
                linewidth=0.4)

        if ctrl_file is not None:
            rcp_ctrl_file = [f for f in ctrl_file if 'rcp_{}'.format(rcp) in f][0]

            cdf_ctrl = cdo.readCdf(rcp_ctrl_file)
            ctrl_t = cdf_ctrl.variables['time'][:]
            cdf_date = np.arange(start_year + step,
                             start_year + (len(ctrl_t[:]) + 1) ,
                             step) 

            ctrl_vals = cdf_ctrl.variables[plot_var][:] - cdf_ctrl.variables[plot_var][0]
            iunits = cdf_ctrl[plot_var].units
            ctrl_vals = -unit_converter(ctrl_vals, iunits, mass_ounits) * gt2mSLE
            ax.plot(cdf_date[:], ctrl_vals,
                    color=rcp_col_dict[rcp],
                    linestyle='solid',
                    linewidth=lw)


        for m_year in [2100, 2200, 2300, 3000]:
            idx = np.where(np.array(date) == m_year)[0][0]
            m_pctl16 = enspctl16_vals[idx]
            m_pctl84 = enspctl84_vals[idx]
            m_pctl16_v = (enspctl16[0] - enspctl16[idx]) / enspctl16[0] * 100
            m_pctl84_v = (enspctl84[0] - enspctl84[idx]) / enspctl84[0] * 100
            m_ctrl = ctrl_vals[idx]
            print(('Year {}: {:1.2f} - {:1.2f} m SLE'.format(m_year, m_pctl84, m_pctl16)))
            print(('Year {}: {:1.2f} - {:1.2f} percent reduction'.format(m_year, m_pctl84_v, m_pctl16_v)))

            print(('         CTRL {:1.2f} m SLE'.format(m_ctrl)))


    if do_legend:
        legend = ax.legend(loc="upper right",
                           edgecolor='0',
                           bbox_to_anchor=(0, 0, .35, 0.88),
                           bbox_transform=plt.gcf().transFigure)
        legend.get_frame().set_linewidth(0.0)
        legend.get_frame().set_alpha(0.0)

    ax.set_xlabel('Year')
    ax.set_ylabel('$\Delta$(GMSL) (m)')
        
    if time_bounds:
        ax.set_xlim(time_bounds[0], time_bounds[1])

    if bounds:
        ax.set_ylim(bounds[0], bounds[1])

    ymin, ymax = ax.get_ylim()

    ax.yaxis.set_major_formatter(FormatStrFormatter('%1.0f'))
    
    if rotate_xticks:
        ticklabels = ax.get_xticklabels()
        for tick in ticklabels:
                tick.set_rotation(30)
    else:
        ticklabels = ax.get_xticklabels()
        for tick in ticklabels:
            tick.set_rotation(0)
                    
    if title is not None:
            plt.title(title)

    set_size(2.44, 0.86)

    for out_format in out_formats:
        out_file = outfile + '_rcp' + '_'  + plot_var + '.' + out_format
        print("  - writing image %s ..." % out_file)
        fig.savefig(out_file, bbox_inches='tight', dpi=out_res)



def plot_rcp_flux(plot_var=flux_plot_vars):

    fig = plt.figure()
    offset = transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)
    ax = fig.add_subplot(111)
    
    for k, rcp in enumerate(rcp_list[::-1]):

        print(('Reading RCP {} files'.format(rcp)))
        rcp_files = [f for f in ifiles if 'rcp_{}'.format(rcp) in f]

        pctl16_file = [f for f in rcp_files if 'enspctl16' in f][0]
        pctl84_file = [f for f in rcp_files if 'enspctl84' in f][0]
            
        cdf_enspctl16 = cdo.runmean('11',input=pctl16_file, returnCdf=True, options=pthreads)
        cdf_enspctl84 = cdo.runmean('11',input=pctl84_file, returnCdf=True, options=pthreads)
        t = cdf_enspctl16.variables['time'][:]

        enspctl16_vals = cdf_enspctl16.variables[plot_var][:]
        iunits = cdf_enspctl16[plot_var].units
        enspctl16_vals = -unit_converter(enspctl16_vals, iunits, flux_ounits) * gt2mmSLE

        enspctl84_vals = cdf_enspctl84.variables[plot_var][:]
        iunits = cdf_enspctl84[plot_var].units
        enspctl84_vals = -unit_converter(enspctl84_vals, iunits, flux_ounits) * gt2mmSLE

        date = np.arange(start_year + step,
                         start_year + (len(t[:]) + 1) ,
                         step) 


        # ensemble between 16th and 84th quantile
        ax.fill_between(date[:], enspctl16_vals, enspctl84_vals,
                        color=rcp_col_dict[rcp],
                        alpha=0.4,
                        linewidth=0)


        ax.plot(date[:], enspctl16_vals,
                color=rcp_col_dict[rcp],
                linestyle='solid',
                linewidth=0.4)

        ax.plot(date[:], enspctl84_vals,
                color=rcp_col_dict[rcp],
                linestyle='solid',
                linewidth=0.4)

        if ctrl_file is not None:
            rcp_ctrl_file = [f for f in ctrl_file if 'rcp_{}'.format(rcp) in f][0]

            cdf_ctrl = cdo.runmean('11', input=rcp_ctrl_file, returnCdf=True, options=pthreads)
            ctrl_t = cdf_ctrl.variables['time'][:]
            ctrl_date = np.arange(start_year + step,
                                 start_year + (len(ctrl_t[:]) + 1) , step) 

            ctrl_vals = cdf_ctrl.variables[plot_var][:]
            iunits = cdf_ctrl[plot_var].units
            ctrl_vals = -unit_converter(ctrl_vals, iunits, flux_ounits) * gt2mmSLE
            ax.plot(ctrl_date[:], ctrl_vals,
                    color=rcp_col_dict[rcp],
                    linestyle='solid',
                    linewidth=lw)


        for m_year in [2100, 2200, 2300]:
            idx = np.where(np.array(date) == m_year)[0][0]
            m_pctl16 = enspctl16_vals[idx]
            m_pctl84 = enspctl84_vals[idx]
            print(('Year {}: {:1.2f} - {:1.2f} mm SLE year-1'.format(m_year, m_pctl84, m_pctl16)))

        idx = np.argmax(enspctl16_vals)
        m_val = enspctl16_vals[idx]
        print(('Max loss rate 16th pctl in Year {}: {:1.3f} mm SLE year-1'.format(m_year, m_val)))
        idx = np.argmax(enspctl84_vals)
        m_year = date[idx]
        m_val = enspctl84_vals[idx]
        print(('Max loss rate 84th pctl in Year {}: {:1.3f} mm SLE year-1'.format(m_year, m_val)))
        idx = np.argmax(enspctl84_vals)
        m_year = ctrl_date[idx]
        m_val = ctrl_vals[idx]
        print(('Max loss rate ctrl in Year {}: {:1.3f} mm SLE year-1'.format(m_year, m_val)))


    if do_legend:
        legend = ax.legend(loc="upper right",
                           edgecolor='0',
                           bbox_to_anchor=(0, 0, .35, 0.88),
                           bbox_transform=plt.gcf().transFigure)
        legend.get_frame().set_linewidth(0.0)
        legend.get_frame().set_alpha(0.0)
                    
    
    ax.set_xlabel('Year')
    ax.set_ylabel('Rate of GMSL rise\n(mm yr$^{\mathregular{-1}}$)')
        
    if time_bounds:
        ax.set_xlim(time_bounds[0], time_bounds[1])

    if bounds:
        ax.set_ylim(bounds[0], bounds[1])

    ymin, ymax = ax.get_ylim()

    ax.yaxis.set_major_formatter(FormatStrFormatter('%1.0f'))

    if rotate_xticks:
        ticklabels = ax.get_xticklabels()
        for tick in ticklabels:
                tick.set_rotation(30)
    else:
        ticklabels = ax.get_xticklabels()
        for tick in ticklabels:
            tick.set_rotation(0)
                    
    if title is not None:
            plt.title(title)

    set_size(2.44, 0.86)

    for out_format in out_formats:
        out_file = outfile + '_rcp' + '_'  + plot_var + '.' + out_format
        print("  - writing image %s ..." % out_file)
        fig.savefig(out_file, bbox_inches='tight', dpi=out_res)

def plot_rcp_d(plot_var=flux_plot_vars):

    fig = plt.figure()
    offset = transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)
    ax = fig.add_subplot(111)

    
    for k, rcp in enumerate(rcp_list[::-1]):

        print(('Reading RCP {} files'.format(rcp)))
        rcp_ctrl_file = [f for f in ifiles if 'rcp_{}'.format(rcp) in f]

        cdf_ctrl = cdo.runmean('11', input=rcp_ctrl_file, returnCdf=True, options=pthreads)
        ctrl_t = cdf_ctrl.variables['time'][:]
        ctrl_date = np.arange(start_year + step,
                             start_year + (len(ctrl_t[:]) + 1) , step) 

        ctrl_vals = cdf_ctrl.variables[plot_var][:]
        iunits = cdf_ctrl[plot_var].units
        ctrl_vals = -unit_converter(ctrl_vals, iunits, flux_ounits) * gt2mmSLE
        ax.plot(ctrl_date[:], ctrl_vals,
                color=rcp_col_dict[rcp],
                linestyle='solid',
                linewidth=lw)

    if do_legend:
        legend = ax.legend(loc="upper right",
                           edgecolor='0',
                           bbox_to_anchor=(0, 0, .35, 0.88),
                           bbox_transform=plt.gcf().transFigure)
        legend.get_frame().set_linewidth(0.0)
        legend.get_frame().set_alpha(0.0)
                    
    
    ax.set_xlabel('Year')
    ax.set_ylabel('Rate (Gt yr$^{\mathregular{-1}}$)')
    ax.set_ylabel('Rate of GMSL eqivalent\n(mm yr$^{\mathregular{-1}}$)')
        
    if time_bounds:
        ax.set_xlim(time_bounds[0], time_bounds[1])

    if bounds:
        ax.set_ylim(bounds[0], bounds[1])

    ymin, ymax = ax.get_ylim()

    ax.yaxis.set_major_formatter(FormatStrFormatter('%1.1f'))

    ax.set_xticks([start_year, 2200, 2400, 2600, 2800, 3000])
    # axGt =  ax.twinx()
    # ymi, yma = ax.get_ylim()
    # axGt.set_ylim(ymi / gt2mmSLE, yma / gt2mmSLE)
    # axGt.set_ylabel('(Gt yr$^{\mathregular{-1}}$')
    
    if rotate_xticks:
        ticklabels = ax.get_xticklabels()
        for tick in ticklabels:
                tick.set_rotation(30)
    else:
        ticklabels = ax.get_xticklabels()
        for tick in ticklabels:
            tick.set_rotation(0)
                    
    if title is not None:
            plt.title(title)

    set_size(2.44, 0.86, ax)
    
    for out_format in out_formats:
        out_file = outfile + '_rcp' + '_'  + plot_var + '.' + out_format
        print("  - writing image %s ..." % out_file)
        fig.savefig(out_file, bbox_inches='tight', dpi=out_res)



def plot_flux_partitioning():

    fig, axa = plt.subplots(4, 3, sharex='col', sharey='row', figsize=[6, 4])
    fig.subplots_adjust(hspace=0.05, wspace=0.05)
    
    for k, rcp in enumerate(rcp_list):
        if rcp == '26':
            m = 0
        elif rcp == '45':
            m = 1
        else:
            m = 2

        rcp_ctrl_file = [f for f in ifiles if 'rcp_{}'.format(rcp) in f and 'CTRL' in f][0]
        rcp_ntrl_file = [f for f in ifiles if 'rcp_{}'.format(rcp) in f and 'NTRL' in f in f][0]
        
        cdf = cdo.runmean('11', input=rcp_ctrl_file, returnCdf=True, options=pthreads)
        cdf_ntrl = cdo.runmean('11', input=rcp_ntrl_file, returnCdf=True, options=pthreads)
        cdf_cum = cdo.timcumsum(input=rcp_ctrl_file, returnCdf=True, options=pthreads)
        t = cdf.variables['time'][:]
        date = np.arange(start_year + step,
                              start_year + (len(t[:]) + 1) , step) 
        t_cum = cdf_cum.variables['time'][:]
        date_cum = np.arange(start_year + step,
                              start_year + (len(t_cum[:]) + 1) , step) 

        area_var = 'ice_area_glacierized'
        area_vals = cdf.variables[area_var][:]
        area_iunits = cdf[area_var].units

        tom_var = 'dMdt'
        tom_vals = np.squeeze(cdf.variables[tom_var][:])
        tom_cum_vals = np.squeeze(cdf_cum.variables[tom_var][:])
        tom_s_vals = tom_vals / area_vals
        tom_iunits = cdf[tom_var].units
        tom_vals = unit_converter(tom_vals, tom_iunits, flux_ounits)
        tom_s_iunits = cf_units.Unit(tom_iunits) / cf_units.Unit(area_iunits)
        tom_s_vals = tom_s_iunits.convert(tom_s_vals, specific_flux_ounits) 
        tom_cum_iunits = cdf[tom_var].units * cf_units.Unit('yr')
        tom_cum_vals = unit_converter(tom_cum_vals, tom_cum_iunits, mass_ounits) * gt2mSLE
        
        snow_var = 'surface_accumulation_rate'
        snow_vals = np.squeeze(cdf.variables[snow_var][:])
        snow_cum_vals = np.squeeze(cdf_cum.variables[snow_var][:])
        snow_s_vals = snow_vals / area_vals
        snow_iunits = cdf[snow_var].units
        snow_vals = unit_converter(snow_vals, snow_iunits, flux_ounits)
        snow_s_iunits = cf_units.Unit(snow_iunits) / cf_units.Unit(area_iunits)
        snow_s_vals = snow_s_iunits.convert(snow_s_vals, specific_flux_ounits) 
        snow_cum_iunits = cdf[snow_var].units * cf_units.Unit('yr')
        snow_cum_vals = unit_converter(snow_cum_vals, snow_cum_iunits, mass_ounits) * gt2mSLE

        ru_var = 'surface_runoff_rate'
        ru_vals = np.squeeze(cdf.variables[ru_var][:])
        ru_ntrl_vals = np.squeeze(cdf_ntrl.variables[ru_var][:])
        ru_cum_vals = cdf_cum.variables[ru_var][:]
        ru_s_vals = ru_vals / area_vals
        ru_ntrl_s_vals = ru_ntrl_vals / area_vals
        ru_iunits = cdf[ru_var].units
        ru_vals = -unit_converter(ru_vals, ru_iunits, flux_ounits)
        ru_ntrl_iunits = cdf_ntrl[ru_var].units
        ru_ntrl_vals = -unit_converter(ru_ntrl_vals, ru_iunits, flux_ounits)
        ru_s_iunits = cf_units.Unit(ru_iunits) / cf_units.Unit(area_iunits)
        ru_s_vals = -ru_s_iunits.convert(ru_s_vals, specific_flux_ounits) 
        ru_ntrl_s_vals = -ru_s_iunits.convert(ru_ntrl_s_vals, specific_flux_ounits) 
        ru_cum_iunits = cdf[ru_var].units * cf_units.Unit('yr')
        ru_cum_vals = unit_converter(ru_cum_vals, ru_cum_iunits, mass_ounits) * gt2mSLE

        d_var = 'tendency_of_ice_mass_due_to_discharge'
        d_vals = np.squeeze(cdf.variables[d_var][:])
        d_cum_vals = np.squeeze(cdf_cum.variables[d_var][:])
        d_s_vals = d_vals / area_vals
        d_iunits = cdf[d_var].units
        d_vals = unit_converter(d_vals, d_iunits, flux_ounits)
        d_s_iunits = cf_units.Unit(d_iunits) / cf_units.Unit(area_iunits)
        d_s_vals = d_s_iunits.convert(d_s_vals, specific_flux_ounits)
        d_cum_iunits = cdf[d_var].units * cf_units.Unit('yr')
        d_cum_vals = unit_converter(d_cum_vals, d_cum_iunits, mass_ounits) * gt2mSLE

        b_var = 'tendency_of_ice_mass_due_to_basal_mass_flux'
        b_vals = np.squeeze(cdf.variables[b_var][:])
        b_cum_vals = np.squeeze(cdf_cum.variables[b_var][:])
        b_s_vals = b_vals / area_vals
        b_iunits = cdf[b_var].units
        b_vals = unit_converter(b_vals, b_iunits, flux_ounits)
        b_s_iunits = cf_units.Unit(b_iunits) / cf_units.Unit(area_iunits)
        b_s_vals = b_s_iunits.convert(b_s_vals, specific_flux_ounits)

        la, = axa[0, m].plot(date, area_vals / 1e12, color='#084594', label='area')
        axa[0, m].set_aspect(200, anchor='S', adjustable='box-forced')
        axa[0, m].set_title('{}'.format(rcp_dict[rcp]))

        # Don't plot basal mass balance
        b_vals = b_s_vals = 0
        
        lsn = axa[1, m].fill_between(date, 0, snow_vals, color='#6baed6', label='accumulation', linewidth=0)
        lruw = axa[1, m].fill_between(date, b_vals, b_vals + ru_vals, color='#fb6a4a', label='runoff (elevation)', linewidth=0)
        lrul = axa[1, m].fill_between(date, b_vals, b_vals + ru_ntrl_vals, color='#fdae6b', label='runoff (climate)', linewidth=0)
        ld = axa[1, m].fill_between(date, b_vals + ru_vals, b_vals + ru_vals + d_vals, color='#74c476', label='discharge', linewidth=0)

        axa[1, m].axhline(0, color='k', linestyle='dotted')
        axa[1, m].plot(date, snow_vals, color='#2171b5', linewidth=0.3)
        axa[1, m].plot(date, b_vals + ru_ntrl_vals, color='#e6550d', linewidth=0.3)
        axa[1, m].plot(date, b_vals + ru_vals, color='#cb181d', linewidth=0.3)
        axa[1, m].plot(date, b_vals + ru_vals + d_vals, color='#238b45', linewidth=0.3)
        lmb, = axa[1, m].plot(date, tom_vals, color='k', label='mass balance', linewidth=0.6)

        lsn = axa[2, m].fill_between(date, 0, snow_s_vals, color='#6baed6', label='accumulation', linewidth=0)
        lruw = axa[2, m].fill_between(date, b_s_vals, b_s_vals + ru_s_vals, color='#fb6a4a', label='runoff (elevation)', linewidth=0)
        lrul = axa[2, m].fill_between(date, b_s_vals, b_s_vals + ru_ntrl_s_vals, color='#fdae6b', label='runoff (climate)', linewidth=0)
        ld = axa[2, m].fill_between(date, b_s_vals + ru_s_vals, b_s_vals + ru_s_vals + d_s_vals, color='#74c476', label='discharge', linewidth=0)
        
        axa[2, m].axhline(0, color='k', linestyle='dotted')
        axa[2, m].plot(date, snow_s_vals, color='#2171b5', linewidth=0.3)
        # axa[2, m].plot(date, b_s_vals, color='#54278f', linewidth=0.3)
        axa[2, m].plot(date, b_s_vals + ru_ntrl_s_vals, color='#e6550d', linewidth=0.3)
        axa[2, m].plot(date, b_s_vals + ru_s_vals, color='#cb181d', linewidth=0.3)
        axa[2, m].plot(date, b_s_vals + ru_s_vals + d_s_vals, color='#238b45', linewidth=0.3)
        lmb, = axa[2, m].plot(date, tom_s_vals, color='k', label='mass balance', linewidth=0.6)

        axa[3, m].axhline(100, color='k', linestyle='dotted')
        axa[3, m].plot(date, -ru_vals / snow_vals * 100, color='#cb181d', label='runoff (total)', linewidth=0.4)
        axa[3, m].plot(date, -d_vals / snow_vals * 100, color='#238b45', label='discharge', linewidth=0.4)
        axa[3, m].plot(date, -tom_vals / snow_vals * 100, color='#000000', label='mass balance', linewidth=0.6)

        axa[3, m].set_xlabel('Year')
        
    axa[0, 0].set_ylabel('Area\n(10$^{6}$ km$^{\mathregular{2}}$)')
    axa[1, 0].set_ylabel('Rate\n(Gt yr$^{\mathregular{-1}}$)')
    axa[2, 0].set_ylabel('Rate\n(kg m$^{\mathregular{-2}}$ yr$^{\mathregular{-1}}$)')
    axa[3, 0].set_ylabel('Ratio\n(%)')
    axm =  axa[2, 2].twinx()
    ymi, yma = axa[2, 2].get_ylim()
    print(ymi, yma)
    axm.set_ylim(ymi / 910., yma / 910.)
    axm.set_yticks([-6, -4, -2, 0, 2])
    axm.set_ylabel('(m yr$^{\mathregular{-1}}$ ice equiv.)')
    
    legend = axa[0, 0].legend(handles=[la],
                              loc="lower left",
                              ncol=1,
                              labelspacing=0.01,
                              handlelength=1.5,
                              columnspacing=1,
                              edgecolor='0',
                              bbox_to_anchor=(.205, 0.72, 0, 0),
                              bbox_transform=plt.gcf().transFigure)
    legend.get_frame().set_linewidth(0.0)
    legend.get_frame().set_alpha(0.0)

    legend = axa[2, 0].legend(handles=[lsn, lrul, lruw, ld, lmb],
                              loc="lower left",
                              ncol=1,
                              labelspacing=0.08,
                              handlelength=1.5,
                              columnspacing=1,
                              edgecolor='0',
                              bbox_to_anchor=(.205, 0.48, 0, 0),
                              bbox_transform=plt.gcf().transFigure)
    legend.get_frame().set_linewidth(0.0)
    legend.get_frame().set_alpha(0.0)

    legend = axa[3, 0].legend(loc="lower left",
                              ncol=1,
                              labelspacing=0.1,
                              handlelength=1.5,
                              columnspacing=1,
                              edgecolor='0',
                              bbox_to_anchor=(.205, 0.18, 0, 0),
                              bbox_transform=plt.gcf().transFigure)
    legend.get_frame().set_linewidth(0.0)
    legend.get_frame().set_alpha(0.0)

    if time_bounds:
        for o in range(0, 3):
            for p in range(0, 3):
                axa[o, p].set_xlim(time_bounds[0], time_bounds[1])

    # ax.yaxis.set_major_formatter(FormatStrFormatter('%1.0f'))

    add_inner_title(axa[0, 0], 'a', 'lower left')
    add_inner_title(axa[0, 1], 'b', 'lower left')
    add_inner_title(axa[0, 2], 'c', 'lower left')
    add_inner_title(axa[1, 0], 'd', 'lower left')
    add_inner_title(axa[1, 1], 'e', 'lower left')
    add_inner_title(axa[1, 2], 'f', 'lower left')
    add_inner_title(axa[2, 0], 'g', 'lower left')
    add_inner_title(axa[2, 1], 'h', 'lower left')
    add_inner_title(axa[2, 2], 'i', 'lower left')
    add_inner_title(axa[3, 0], 'j', 'upper left')
    add_inner_title(axa[3, 1], 'k', 'upper left')
    add_inner_title(axa[3, 2], 'l', 'upper left')

    if rotate_xticks:
        for o, p in list(range(0, 2)), list(range(0, 2)):
            ticklabels = axa[o, p].get_xticklabels()
            for tick in ticklabels:
                tick.set_rotation(30)
    else:
        for o, p in list(range(0, 2)), list(range(0, 2)):
            ticklabels = axa[o, p].get_xticklabels()
            for tick in ticklabels:
                tick.set_rotation(0)

    # if title is not None:
    #     plt.title(title)

    
    for out_format in out_formats:
        out_file = outfile + '_partitioning.' + out_format
        print("  - writing image %s ..." % out_file)
        fig.savefig(out_file, bbox_inches='tight', dpi=out_res)

        
def plot_basin_flux_partitioning():

    fig, axa = plt.subplots(6, 3, sharex='col', sharey='row', figsize=[6, 4])
    fig.subplots_adjust(hspace=0.06, wspace=0.04)
    
    for k, rcp in enumerate(rcp_list):
        if rcp == '26':
            m = 0
        elif rcp == '45':
            m = 1
        else:
            m = 2

        for k, basin in enumerate(basin_list):

            basin_files = [f for f in ifiles if 'b_{}'.format(basin) in f]
    
            rcp_ctrl_file = [f for f in basin_files if 'rcp_{}'.format(rcp) in f and 'CTRL' in f][0]
            # rcp_ntrl_file = [f for f in basin_files if 'rcp_{}'.format(rcp) in f and 'NTRL' in f in f][0]
            print(('Reading {}'.format( rcp_ctrl_file)))
            # print('Reading {}'.format( rcp_ntrl_file))
            cdf = cdo.runmean('11', input=rcp_ctrl_file, returnCdf=True, options=pthreads)
            # cdf_ntrl = cdo.runmean('11', input=rcp_ntrl_file, returnCdf=True, options=pthreads)

            t = cdf.variables['time'][:]
            date = np.arange(start_year + step,
                             start_year + (len(t[:]) + 1) , step) 

            tom_var = 'dMdt'
            tom_vals = np.squeeze(cdf.variables[tom_var][:])
            tom_iunits = cdf[tom_var].units
            tom_vals = unit_converter(tom_vals, tom_iunits, flux_ounits)

            snow_var = 'surface_accumulation_rate'
            snow_vals = np.squeeze(cdf.variables[snow_var][:])
            snow_iunits = cdf[snow_var].units
            snow_vals = unit_converter(snow_vals, snow_iunits, flux_ounits)

            ru_var = 'surface_runoff_rate'
            ru_vals = np.squeeze(cdf.variables[ru_var][:])
            # ru_ntrl_vals = np.squeeze(cdf_ntrl.variables[ru_var][:])
            ru_iunits = cdf[ru_var].units
            ru_vals = -unit_converter(ru_vals, ru_iunits, flux_ounits)
            # ru_ntrl_iunits = cdf_ntrl[ru_var].units
            # ru_ntrl_vals = -unit_converter(ru_ntrl_vals, ru_iunits, flux_ounits)

            d_var = 'tendency_of_ice_mass_due_to_discharge'
            d_vals = np.squeeze(cdf.variables[d_var][:])
            d_iunits = cdf[d_var].units
            d_vals = unit_converter(d_vals, d_iunits, flux_ounits)
            
            lsn = axa[k,m].fill_between(date, 0, snow_vals, color='#6baed6', label='accumulation', linewidth=0)
            lruw = axa[k,m].fill_between(date, 0, ru_vals, color='#fb6a4a', label='runoff', linewidth=0)
            # lrul = axa[k,m].fill_between(date, 0, ru_ntrl_vals, color='#fdae6b', label='RW', linewidth=0)
            ld = axa[k,m].fill_between(date, ru_vals, ru_vals + d_vals, color='#74c476', label='discharge', linewidth=0)
            axa[k,m].plot(date, snow_vals, color='#2171b5', linewidth=0.3)
            axa[k,m].plot(date, ru_vals, color='#cb181d', linewidth=0.3)
            #axa[k,m].plot(date, ru_ntrl_vals, color='#e6550d', linewidth=0.3)
            axa[k,m].plot(date, ru_vals + d_vals, color='#238b45', linewidth=0.3)
            lmb, = axa[k,m].plot(date, tom_vals, color='k', label='mass balance', linewidth=0.6)
            axa[k,m].axhline(0, color='k', linestyle='dotted')
            
            axa[k,m].yaxis.set_major_formatter(FormatStrFormatter('%1.0f'))

            if k == 5:
                axa[k,m].set_xlabel('Year')
            if m == 0:
                axa[k,m].set_ylabel('Rate\n (Gt yr$^{\mathregular{-1}}$)')

            if time_bounds:
                axa[k,m].set_xlim(time_bounds[0], time_bounds[1])

            if bounds:
                axa[k,m].set_ylim(bounds[0], bounds[1])
        
            if rotate_xticks:
                ticklabels = axa[k,m].get_xticklabels()
                for tick in ticklabels:
                        tick.set_rotation(30)
            else:
                ticklabels = axa[k,m].get_xticklabels()
                for tick in ticklabels:
                    tick.set_rotation(0)
                    
    legend = axa[0, 2].legend(handles=[lsn, lruw, ld, lmb],
                              loc="upper right",
                              ncol=1,
                              labelspacing=0.1,
                              handlelength=1.5,
                              columnspacing=1,
                              edgecolor='0',
                              bbox_to_anchor=(.45, 0.075, 0, 0),
                              bbox_transform=plt.gcf().transFigure)
    legend.get_frame().set_linewidth(0.0)
    legend.get_frame().set_alpha(0.0)
    
    for out_format in out_formats:
        out_file = outfile + '_basin_partitioning.' + out_format
        print("  - writing image %s ..." % out_file)
        fig.savefig(out_file, bbox_inches='tight', dpi=out_res)

        
def plot_basin_cumulative_partitioning():

    width = 0.2
    dist = 0.2

    fig, axa = plt.subplots(6, 3, sharex='col', sharey='row', figsize=[6, 4])
    fig.subplots_adjust(hspace=0.06, wspace=0.04)

    fig_bar, axb = plt.subplots(2, 1, sharex='col')

    scale_factor = 1
    for m, rcp in enumerate(rcp_list):
        if rcp == '26':
            m = 0
            m_years = [2316, 2918]
        elif rcp == '45':
            m = 1
            m_years = [2189, 2556]
        else:
            m = 2
            m_years = [2137, 2202]

        for k, basin in enumerate(basin_list):

            basin_files = [f for f in ifiles if 'b_{}'.format(basin) in f]
    
            rcp_ctrl_file = [f for f in basin_files if 'rcp_{}'.format(rcp) in f and 'CTRL' in f][0]
            # rcp_ntrl_file = [f for f in basin_files if 'rcp_{}'.format(rcp) in f and 'NTRL' in f in f][0]
            print(('Reading {}'.format( rcp_ctrl_file)))
            # print('Reading {}'.format( rcp_ntrl_file))
            cdf = cdo.timcumsum(input=rcp_ctrl_file, returnCdf=True, options=pthreads)
            # cdf_ntrl = cdo.runmean('11', input=rcp_ntrl_file, returnCdf=True, options=pthreads)

            time_units = 'year'

            t = cdf.variables['time'][:]
            date = np.arange(start_year + step,
                             start_year + (len(t[:]) + 1) , step) 

            tom_var = 'tendency_of_ice_mass'
            tom_vals = np.squeeze(cdf.variables[tom_var][:])
            tom_iunits = cdf[tom_var].units
            iunits_cf = cf_units.Unit(tom_iunits) * cf_units.Unit(time_units)
            ounits_cf = cf_units.Unit(mass_ounits)
            tom_vals = iunits_cf.convert(tom_vals, ounits_cf) / scale_factor

            snow_var = 'surface_accumulation_rate'
            snow_vals = np.squeeze(cdf.variables[snow_var][:])
            snow_iunits = cdf[snow_var].units
            iunits_cf = cf_units.Unit(snow_iunits) * cf_units.Unit(time_units)
            snow_vals = iunits_cf.convert(snow_vals, ounits_cf) / scale_factor

            ru_var = 'surface_runoff_rate'
            ru_vals = -np.squeeze(cdf.variables[ru_var][:])
            # ru_ntrl_vals = np.squeeze(cdf_ntrl.variables[ru_var][:])
            ru_iunits = cdf[ru_var].units
            iunits_cf = cf_units.Unit(ru_iunits) * cf_units.Unit(time_units)
            ru_vals = iunits_cf.convert(ru_vals, ounits_cf) / scale_factor
            # ru_ntrl_iunits = cdf_ntrl[ru_var].units
            # ru_ntrl_vals = -unit_converter(ru_ntrl_vals, ru_iunits, mass_ounits)

            d_var = 'tendency_of_ice_mass_due_to_discharge'
            d_vals = np.squeeze(cdf.variables[d_var][:])
            d_iunits = cdf[d_var].units
            iunits_cf = cf_units.Unit(d_iunits) * cf_units.Unit(time_units)
            d_vals = iunits_cf.convert(d_vals, ounits_cf) / scale_factor

            axb[0].bar(k+m*dist-dist, snow_vals[m_years[0]-start_year], width=width, color='#6baed6')
            axb[0].bar(k+m*dist-dist, ru_vals[m_years[0]-start_year], width=width, color='#fb6a4a')
            axb[0].bar(k+m*dist-dist, d_vals[m_years[0]-start_year], bottom=ru_vals[m_years[0]-start_year], width=width, color='#74c476')
            
            axb[1].bar(k+m*dist-dist, snow_vals[m_years[1]-start_year], width=width, color='#6baed6')
            axb[1].bar(k+m*dist-dist, ru_vals[m_years[1]-start_year], width=width, color='#fb6a4a')
            axb[1].bar(k+m*dist-dist, d_vals[m_years[1]-start_year], bottom=ru_vals[m_years[1]-start_year], width=width, color='#74c476')
            

    axb[0].axhline(0, color='k', linewidth=0.3, linestyle='dashed')
    axb[1].axhline(0, color='k', linewidth=0.3, linestyle='dashed')
    plt.xticks([0,1,2,3,4,5],basin_list)

     
    for out_format in out_formats:
        out_file = outfile + '_basin_partitioning_bar.' + out_format
        print("  - writing image %s ..." % out_file)
        fig_bar.savefig(out_file, bbox_inches='tight', dpi=out_res)



def plot_rcp_flux_cumulative(plot_var=flux_plot_vars):
    
    fig = plt.figure()
    offset = transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)
    ax = fig.add_subplot(111)
    
    for k, rcp in enumerate(rcp_list):

        rcp_files = [f for f in ifiles if 'rcp_{}'.format(rcp) in f]
        if len(rcp_files) < 3:
            
            print(('Less than 3 files found for {}, skipping'.format(rcp_dict[rcp])))

        else:

            print(('Reading files for {}'.format(rcp_dict[rcp])))
            
            cdf_enspctl16 = cdo.enspctl('16',input=rcp_files, options=pthreads)
            cdf_enspctl16 = cdo.timcumsum(input=cdf_enspctl16, returnCdf=True, options=pthreads)
            cdf_enspctl84 = cdo.enspctl('84',input=rcp_files, options=pthreads)
            cdf_enspctl84 = cdo.timcumsum(input=cdf_enspctl84, returnCdf=True, options=pthreads)
            t = cdf_enspctl16.variables['time'][:]

            enspctl16_vals = cdf_enspctl16.variables[plot_var][:]
            iunits = cdf_enspctl16[plot_var].units
            iunits_cf = cf_units.Unit(iunits) * cf_units.Unit('yr')
            o_units_cf = cf_units.Unit(mass_ounits)
            enspctl16_vals = iunits_cf.convert(enspctl16_vals, o_units_cf) * gt2mSLE

            enspctl84_vals = cdf_enspctl84.variables[plot_var][:]
            iunits = cdf_enspctl84[plot_var].units
            iunits_cf = cf_units.Unit(iunits) * cf_units.Unit('yr')
            o_units_cf = cf_units.Unit(mass_ounits)
            enspctl84_vals = iunits_cf.convert(enspctl84_vals, o_units_cf) * gt2mSLE
            
            
            date = np.arange(start_year + step,
                             start_year + (len(t[:]) + 1) ,
                             step) 


            # ensemble between 16th and 84th quantile
            ax.fill_between(date[:], enspctl16_vals, enspctl84_vals,
                            color=rcp_col_dict[rcp],
                            alpha=0.4,
                            linewidth=0)

            ax.plot(date[:], ensmedian_vals,
                    color=rcp_col_dict[rcp],
                    linewidth=lw,
                    label=rcp_dict[rcp])

            ax.plot(date[:], enspctl16_vals,
                    color=rcp_col_dict[rcp],
                    linestyle='solid',
                    linewidth=0.25)

            ax.plot(date[:], enspctl84_vals,
                    color=rcp_col_dict[rcp],
                    linestyle='solid',
                    linewidth=0.25)

            for m_year in [2100, 2200, 2300]:
                idx = np.where(np.array(date) == m_year)[0][0]
                m_median = ensmedian_vals[idx]
                m_pctl84 = enspctl84_vals[idx]
                print(('Year {}: {:1.2f} - {:1.2f} - {:1.2f} mm SLE year-1'.format(m_year, m_pctl84, m_median, m_pctl16)))


    if do_legend:
        legend = ax.legend(loc="upper right",
                           edgecolor='0',
                           bbox_to_anchor=(0, 0, .35, 0.88),
                           bbox_transform=plt.gcf().transFigure)
        legend.get_frame().set_linewidth(0.0)
        legend.get_frame().set_alpha(0.0)

    
    ax.set_xlabel('Year')
    ax.set_ylabel('$\Delta$(GMSL) (cm yr$^{\mathregular{-1}}$)')
        
    if time_bounds:
        ax.set_xlim(time_bounds[0], time_bounds[1])

    if bounds:
        ax.set_ylim(bounds[0], bounds[1])

    ymin, ymax = ax.get_ylim()

    ax.yaxis.set_major_formatter(FormatStrFormatter('%1.2f'))

    if rotate_xticks:
        ticklabels = ax.get_xticklabels()
        for tick in ticklabels:
                tick.set_rotation(30)
    else:
        ticklabels = ax.get_xticklabels()
        for tick in ticklabels:
            tick.set_rotation(0)
                    
    if title is not None:
            plt.title(title)

    for out_format in out_formats:
        out_file = outfile + '_rcp' + '_'  + plot_var + '.' + out_format
        print("  - writing image %s ..." % out_file)
        fig.savefig(out_file, bbox_inches='tight', dpi=out_res)



def plot_basin_mass():
    
    fig = plt.figure()
    offset = transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)
    ax = fig.add_subplot(111)

    mass_var_vals_positive_cum = 0
    mass_var_vals_negative_cum = 0
    for k, ifile in enumerate(ifiles):
        basin = basin_list[k]
        print(('reading {}'.format(ifile)))
        nc = NC(ifile, 'r')
        t = nc.variables["time"][:]

        date = np.arange(start_year + step,
                         start_year + (len(t[:]) + 1) ,
                         step) 

        idx = np.where(np.array(date) == time_bounds[-1])[0][0]
        mvar = 'ice_mass'
        mass_var_vals = -np.squeeze(nc.variables[mvar][:] - nc.variables[mvar][0]) * gt2mSLE
        iunits = nc.variables[mvar].units
        mass_var_vals = unit_converter(mass_var_vals, iunits, mass_ounits)
        if mass_var_vals[idx] > 0:
            ax.fill_between(date[:], mass_var_vals_positive_cum, mass_var_vals_positive_cum + mass_var_vals[:],
                            color=basin_col_dict[basin],
                            linewidth=0,
                            label=basin)
        else:
            ax.fill_between(date[:], mass_var_vals_negative_cum, mass_var_vals_negative_cum + mass_var_vals[:],
                            color=basin_col_dict[basin],
                            linewidth=0,
                            label=basin)
            plt.rcParams['hatch.color'] = basin_col_dict[basin]
            plt.rcParams['hatch.linewidth'] = 0.1
            ax.fill_between(date[:], mass_var_vals_negative_cum, mass_var_vals_negative_cum + mass_var_vals[:],
                            facecolor="none", hatch="XXXXX", edgecolor="k",
                            linewidth=0.0)

        if mass_var_vals[idx] > 0:
            ax.plot(date[:], mass_var_vals_positive_cum + mass_var_vals[:],
                    color='k',
                    linewidth=0.1)
        else:
            ax.plot(date[:], mass_var_vals_negative_cum + mass_var_vals[:],
                    color='k',
                    linewidth=0.1)

        offset = 0
        if mass_var_vals[idx] > 0:
            try:
                x_sle, y_sle = date[idx] + offset, mass_var_vals_positive_cum[idx]
            except:  # first iteration
                x_sle, y_sle = date[idx] + offset, mass_var_vals_positive_cum
        else:
            try:
                x_sle, y_sle = date[idx] + offset, mass_var_vals_negative_cum[idx] + mass_var_vals[idx] 
            except:  # first iteration
                x_sle, y_sle = date[idx] + offset, mass_var_vals_negative_cum + mass_var_vals[idx] 
        nc.close()
        if mass_var_vals[idx] > 0:
            mass_var_vals_positive_cum += mass_var_vals
        else:
            mass_var_vals_negative_cum += mass_var_vals

        print(('Basin {}'.format(basin)))
        for m_year in [2100, 2200, 2500, 3000]:
            idx = np.where(np.array(date) == m_year)[0][0]
            m = mass_var_vals[idx]
            print(('Year {}: {:1.2f} m SLE'.format(m_year, m)))


    ax.hlines(0, time_bounds[0], time_bounds[-1], lw=0.25)

    legend = ax.legend(loc="upper right",
                       edgecolor='0',
                       bbox_to_anchor=(0, 0, 1.15, 1),
                       bbox_transform=plt.gcf().transFigure)
    legend.get_frame().set_linewidth(0.2)
    # legend.get_frame().set_alpha(0.0)

    
    ax.set_xlabel('Year')
    ax.set_ylabel('$\Delta$(GMSL) (m)')
        
    if time_bounds:
        ax.set_xlim(time_bounds[0], time_bounds[1])

    if bounds:
        ax.set_ylim(bounds[0], bounds[-1])

    ax.yaxis.set_major_formatter(FormatStrFormatter('%1.2f'))

    if rotate_xticks:
        ticklabels = ax.get_xticklabels()
        for tick in ticklabels:
                tick.set_rotation(30)
    else:
        ticklabels = ax.get_xticklabels()
        for tick in ticklabels:
            tick.set_rotation(0)
                    
    if title is not None:
            plt.title(title)

    for out_format in out_formats:
        out_file = outfile + '_' + mvar  + '.' + out_format
        print("  - writing image %s ..." % out_file)
        fig.savefig(out_file, bbox_inches='tight', dpi=out_res)  
 
def plot_basin_flux(plot_var='discharge'):
    '''
    Make a plot per basin with all flux_plot_vars
    '''

    fig = plt.figure()
    offset = transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)
    ax = fig.add_subplot(111)

    for basin in basin_list:

        basin_file = [f for f in ifiles if 'b_{}'.format(basin) in f]
        print(basin_file)

        print(('reading {}'.format(basin_file[0])))

        if plot_var == 'discharge':
            cdf = cdo.expr('discharge=tendency_of_ice_mass_due_to_discharge+tendency_of_ice_mass_due_to_basal_mass_flux', input=basin_file[0])
            cdf_run = cdo.runmean('11', input=cdf, returnCdf=True, options=pthreads)
        
            iunits = 'Gt year-1'
            var_vals = cdf_run.variables[plot_var][:]

        t = cdf_run.variables["time"][:]

        date = np.arange(start_year + step,
                         start_year + (len(t[:]) + 1) ,
                         step) 


        var_vals = unit_converter(np.squeeze(var_vals), iunits, flux_ounits)
        plt.plot(date[:], var_vals[:],
                 color=basin_col_dict[basin],
                 lw=lw)

    if do_legend:
        legend = ax.legend(loc="upper right",
                           edgecolor='0',
                           bbox_to_anchor=(0, 0, 1.15, 1),
                           bbox_transform=plt.gcf().transFigure)
        legend.get_frame().set_linewidth(0.2)
    
    ax.set_xlabel('Year')
    ax.set_ylabel('mass flux (Gt yr$^{\mathregular{-1}}$)')
        
    if time_bounds:
        ax.set_xlim(time_bounds[0], time_bounds[1])
            
    if bounds:
        ax.set_ylim(bounds[0], bounds[1])

    if rotate_xticks:
        ticklabels = ax.get_xticklabels()
        for tick in ticklabels:
            tick.set_rotation(30)
    else:
        ticklabels = ax.get_xticklabels()
        for tick in ticklabels:
            tick.set_rotation(0)
                    
    if title is not None:
        plt.title(title)

    for out_format in out_formats:
        out_file = outfile  + '_fluxes.' + out_format
        print("  - writing image %s ..." % out_file)
        fig.savefig(out_file, bbox_inches='tight', dpi=out_res)
                   

def plot_per_basin_flux(plot_var=None):
    '''
    Make a plot per basin with all flux_plot_vars
    '''

    for basin in basin_list:

        fig = plt.figure()
        offset = transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)
        ax = fig.add_subplot(111)

        basin_file = [f for f in ifiles if 'b_{}'.format(basin) in f]

        for k, rcp in enumerate(rcp_list[::-1]):
            rcp_file = [f for f in basin_file if 'rcp_{}'.format(rcp) in f]

            print(('reading {}'.format(rcp_file[0])))

            cdf_run = cdo.runmean('11', input=rcp_file[0], returnCdf=True, options=pthreads)
            t = cdf_run.variables["time"][:]
            date = np.arange(start_year + step,
                             start_year + (len(t[:]) + 1) ,
                             step) 

            if plot_var is None:
                m_vars = ['tendency_of_ice_mass_due_to_surface_mass_flux',
                          'tendency_of_ice_mass_due_to_discharge']
                label_var = 'fluxes'
            else:
                m_vars = plot_var
                label_var = plot_var[-1]
                
            for m_var in m_vars:

                iunits = cdf_run.variables[m_var].units
                var_vals = cdf_run.variables[m_var][:]

                var_vals = unit_converter(np.squeeze(var_vals), iunits, flux_ounits)
                plt.plot(date[:], var_vals[:],
                         color=rcp_col_dict[rcp],
                         #ls=flux_style_dict[m_var],
                         ls='solid',
                         lw=lw)

        if do_legend:
            legend = ax.legend(loc="upper right",
                               edgecolor='0',
                               bbox_to_anchor=(0, 0, 1.15, 1),
                               bbox_transform=plt.gcf().transFigure)
            legend.get_frame().set_linewidth(0.2)
    
        ax.set_xlabel('Year')
        ax.set_ylabel('Rate (Gt yr$^{\mathregular{-1}}$)')
        
        if time_bounds:
            ax.set_xlim(time_bounds[0], time_bounds[1])
            
        if bounds:
            ax.set_ylim(bounds[0], bounds[1])

        if rotate_xticks:
            ticklabels = ax.get_xticklabels()
            for tick in ticklabels:
                tick.set_rotation(30)
        else:
            ticklabels = ax.get_xticklabels()
            for tick in ticklabels:
                tick.set_rotation(0)
                    
        if title is not None:
            plt.title(title)

        for out_format in out_formats:
            out_file = outfile  + '_' + basin + '_' + label_var + '.' + out_format
            print("  - writing image %s ..." % out_file)
            fig.savefig(out_file, bbox_inches='tight', dpi=out_res)
                   
if plot == 'ctrl_mass':
    plot_ctrl_mass(plot_var='limnsw')
elif plot == 'ctrl_mass_anim':
    plot_ctrl_mass_anim(plot_var='limnsw')
if plot == 'percent_mass':
    plot_percent_mass(plot_var='limnsw')
elif plot == 'rcp_mass':
    plot_rcp_mass(plot_var='limnsw')
elif plot == 'rcp_flux':
    plot_rcp_flux(plot_var='tendency_of_ice_mass_glacierized')
elif plot == 'rcp_d':
    plot_rcp_d(plot_var='tendency_of_ice_mass_due_to_discharge')
elif plot == 'rcp_accum':
    plot_rcp_flux_cumulative(plot_var='surface_accumulation_rate')
elif plot == 'rcp_d':
    plot_rcp_flux_cumulative(plot_var='tendency_of_ice_mass_due_to_discharge')
elif plot == 'rcp_traj':
    plot_rcp_traj_mass(plot_var='limnsw')
elif plot == 'basin_mass':
    plot_basin_mass()
elif plot == 'basin_d':
    plot_basin_flux(plot_var='discharge')
elif plot == 'per_basin_flux':
    plot_per_basin_flux(plot_var=['tendency_of_ice_mass_due_to_discharge'])
elif plot == 'per_basin_d':
    plot_per_basin_flux(plot_var='discharge_flux')
elif plot == 'flux_partitioning':
    plot_flux_partitioning()
elif plot == 'basin_flux_partitioning':
    plot_basin_flux_partitioning()
elif plot == 'basin_cumulative_partitioning':
    plot_basin_cumulative_partitioning()
elif plot == 'cmip5':
    plot_cmip5_forcing()
elif plot == 'cmip5_rcp':
    plot_cmip5_rcp()
elif plot == 'station_usurf':
    plot_point_ts()
elif plot == 'grid_res':
    plot_grid_res()
elif plot == 'grid_pc':
    plot_grid_pc()
elif plot == 'profile_combined':
    plot_profile_ts_combined()
