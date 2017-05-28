#!/usr/bin/env python

# Copyright (C) 2011-2015 Andy Aschwanden

import numpy as np
import pylab as plt
from argparse import ArgumentParser
import matplotlib.transforms as transforms
from netCDF4 import Dataset as NC

try:
    from pypismtools import unit_converter, set_mode, get_golden_mean, smooth
except:
    from pypismtools.pypismtools import unit_converter, set_mode, get_golden_mean, smooth

all_basins = ['CW', 'NE', 'NO', 'NW', 'SE', 'SW', 'GR']
# Set up the option parser
parser = ArgumentParser()
parser.description = "A script for PISM output files to time series plots using pylab/matplotlib."
parser.add_argument("FILE", nargs='*')
parser.add_argument("--bounds", dest="bounds", nargs=2, type=float,
                    help="lower and upper bound for ordinate, eg. -1 1", default=None)
parser.add_argument("--time_bounds", dest="time_bounds", nargs=2, type=float,
                    help="lower and upper bound for abscissa, eg. 1990 2000", default=None)
parser.add_argument("-a", "--aspect_ratio", dest="aspect_ratio", type=float,
                    help="Plot aspect ratio", default=0.75)
parser.add_argument("-b", "--basin", dest="basin",
                    choices=all_basins,
                    help="Basin to plot", default='GR')
parser.add_argument("-s", "--switch_sign", dest="switch_sign", action='store_true',
                    help="Switch sign of data", default=False)
parser.add_argument("-l", "--labels", dest="labels",
                    help="comma-separated list with labels, put in quotes like 'label 1,label 2'", default=None)
parser.add_argument("--index_ij", dest="index_ij", nargs=2, type=int,
                    help="i and j index for spatial fields, eg. 10 10", default=[0, 0])
parser.add_argument("--lon_lat", dest="lon_lat", nargs=2, type=float,
                    help="lon and lat for spatial fields, eg. 10 10", default=None)
parser.add_argument("-f", "--output_format", dest="out_formats",
                    help="Comma-separated list with output graphics suffix, default = pdf", default='pdf')
parser.add_argument("-n", "--normalize", dest="normalize", action="store_true",
                    help="Normalize to beginning of time series, Default=False", default=False)
parser.add_argument("-o", "--output_file", dest="outfile",
                    help="output file name without suffix, i.e. ts_control -> ts_control_variable", default='unnamed')
parser.add_argument("-p", "--print_size", dest="print_mode",
                    help="sets figure size and font size, available options are: \
                  'onecol','publish','medium','presentation','twocol'", default="medium")
parser.add_argument("--step", dest="step", type=int,
                    help="step for plotting values, if time-series is very long", default=1)
parser.add_argument("--show", dest="show", action="store_true",
                    help="show figure (in addition to save), Default=False", default=False)
parser.add_argument("--shadow", dest="shadow", action="store_true",
                    help='''add drop shadow to line plots, Default=False''',
                    default=False)
parser.add_argument("--start_year", dest="start_year", type=float,
                    help='''Start year''', default=2008)
parser.add_argument("--rotate_xticks", dest="rotate_xticks", action="store_true",
                    help="rotate x-ticks by 30 degrees, Default=False",
                    default=False)
parser.add_argument("-r", "--output_resolution", dest="out_res",
                    help='''Resolution ofoutput graphics in dots per
                  inch (DPI), default = 300''', default=300)
parser.add_argument("-t", "--twinx", dest="twinx", action="store_true",
                    help='''adds a second ordinate with units mmSLE,
                  Default=False''', default=False)
parser.add_argument("--title", dest="title",
                    help='''Plot title. default=False''', default=None)

options = parser.parse_args()
aspect_ratio = options.aspect_ratio
basin = options.basin
ifiles = options.FILE
if options.labels != None:
    labels = options.labels.split(',')
else:
    labels = None
bounds = options.bounds
time_bounds = options.time_bounds
golden_mean = get_golden_mean()
normalize = options.normalize
out_res = options.out_res
outfile = options.outfile
out_formats = options.out_formats.split(',')
print_mode = options.print_mode
rotate_xticks = options.rotate_xticks
step = options.step
shadow = options.shadow
show = options.show
title = options.title
twinx = options.twinx
dashes = ['-', '--', '-.', ':', '-', '--', '-.', ':']

dx, dy = 4. / out_res, -4. / out_res

# Conversion between giga tons (Gt) and millimeter sea-level equivalent (mmSLE)
gt2mmSLE = 1. / 365

start_year = options.start_year

# Plotting styles
axisbg = '1'
shadow_color = '0.25'
numpoints = 1




# set the print mode
lw, pad_inches = set_mode(print_mode, aspect_ratio=aspect_ratio)

#plt.rcParams['legend.fancybox'] = True

basin_col_dict = {'CW': '#998ec3',
                  'NE': '#fdb863',
                  'NO': '#018571',
                  'NW': '#d8daeb',
                  'SE': '#e66101',
                  'SW': '#542788',
                  'GR': '#000000'}


rcp_col_dict = {'RCP85': '#ca0020',
                'RCP63': '#f4a582',
                'RCP45': '#92c5de',
                'RCP26': '#0571b0'}

rcp_list = ['RCP26', 'RCP45', 'RCP63', 'RCP85']
rcp_dict = {'RCP26': 'RCP 2.6',
            'RCP45': 'RCP 4.5',
            'RCP63': 'RCP 6.3',
            'RCP85': 'RCP 8.5'}

flux_vars = ['mass_rate_of_change_glacierized', 'discharge_flux', 'surface_ice_flux', 'sub_shelf_ice_flux', 'grounded_basal_ice_flux']
flux_abbr_dict = {'mass_rate_of_change_glacierized': '$\dot M$', 'discharge_flux': 'D', 'surface_ice_flux': 'SMB', 'sub_shelf_ice_flux': 'FMB', 'grounded_basal_ice_flux': 'BMB'}
flux_style_dict = {'mass_rate_of_change_glacierized': '-', 'discharge_flux': '--', 'surface_ice_flux': ':', 'sub_shelf_ice_flux': ':', 'grounded_basal_ice_flux': '-.'}
flux_plot_vars = ['mass_rate_of_change_glacierized', 'discharge_flux', 'surface_ice_flux']
mass_plot_vars = ['limnsw']

def plot_fluxes(plot_vars):

    ifile = ifiles[0]
    nc = NC(ifile, 'r')
    t = nc.variables["time"][:]

    date = np.arange(start_year + step,
                 start_year + (len(t[:]) + 1) * step,
                 step)
    
    fig = plt.figure()
    offset = transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)
    ax = fig.add_subplot(111, axisbg=axisbg)
    
    for mvar in plot_vars:
        var_vals = nc.variables[mvar][:] / 1e12
        runmean_var_vals = smooth(var_vals, window_len=10)
        plt.plot(date[2::], var_vals[2::],
                 color=basin_col_dict[basin],
                 lw=0.75,
                 ls=style_dict[mvar],
                 label=abbr_dict[mvar])
        plt.plot(date[2::], runmean_var_vals[2::],
                 color=basin_col_dict[basin],
                 lw=1,
                 ls=style_dict[mvar])

        ax.legend(loc="upper right",
                  shadow=False,
                  bbox_to_anchor=(0, 0, 1, 1),
                  bbox_transform=plt.gcf().transFigure)

        if twinx:
            axSLE = ax.twinx()
            ax.set_autoscalex_on(False)
            axSLE.set_autoscalex_on(False)
        
            ax.set_xlabel('yr')
            ax.set_ylabel('flux (Gt/yr')
            
            if time_bounds:
                ax.set_xlim(time_bounds[0], time_bounds[1])

            if bounds:
                ax.set_ylim(bounds[0], bounds[1])

                ymin, ymax = ax.get_ylim()
            if twinx:
                # Plot twin axis on the right, in mmSLE
                yminSLE = ymin * gt2mmSLE
                ymaxSLE = ymax * gt2mmSLE
                axSLE.set_xlim(date_start, date_end)
                axSLE.set_ylim(yminSLE, ymaxSLE)
                axSLE.set_ylabel(sle_label)

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
        out_file = outfile + '_fluxes'  + '.' + out_format
        print "  - writing image %s ..." % out_file
        fig.savefig(out_file, bbox_inches='tight', dpi=out_res)

def plot_mass(plot_vars=mass_plot_vars):

    
    fig = plt.figure()
    offset = transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)
    ax = fig.add_subplot(111, axisbg=axisbg)

    for k, rcp in enumerate(rcp_list):
        ifile = ifiles[k]
        print('reading {}'.format(ifile))
        nc = NC(ifile, 'r')
        t = nc.variables["time"][:]

        date = np.arange(start_year + step,
                         start_year + (len(t[:]) + 1) * step,
                         step) 

    
        for mvar in plot_vars:
            var_vals = nc.variables[mvar][:]
            # plot anomalies
            plt.plot(date[:], (var_vals[:] - var_vals[0]) / 1e12,
                     color=rcp_col_dict[rcp],
                     lw=0.75,
                     label=rcp_dict[rcp])
    nc.close()

    ax.legend(loc="upper right",
              shadow=False,
              bbox_to_anchor=(0, 0, 1, 1),
              bbox_transform=plt.gcf().transFigure)

    if twinx:
        axSLE = ax.twinx()
        ax.set_autoscalex_on(False)
        axSLE.set_autoscalex_on(False)
        
        ax.set_xlabel('yr')
        ax.set_ylabel('cumulative mass change (Gt)')
        
    if time_bounds:
        ax.set_xlim(time_bounds[0], time_bounds[1])

    if bounds:
        ax.set_ylim(bounds[0], bounds[1])

    ymin, ymax = ax.get_ylim()
    if twinx:
        # Plot twin axis on the right, in mmSLE
        yminSLE = ymin * gt2mmSLE
        ymaxSLE = ymax * gt2mmSLE
        axSLE.set_ylim(yminSLE, ymaxSLE)
        axSLE.set_ylabel('mm SLE')

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
        out_file = outfile + '_mass'  + '.' + out_format
        print "  - writing image %s ..." % out_file
        fig.savefig(out_file, bbox_inches='tight', dpi=out_res)


plot_mass(mass_plot_vars)
#plot_fluxes(plot_vars=flux_plot_vars)
