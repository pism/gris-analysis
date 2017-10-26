#!/usr/bin/env python

# Copyright (C) 2017 Andy Aschwanden

from argparse import ArgumentParser
import matplotlib.transforms as transforms
from matplotlib.ticker import FormatStrFormatter
from netCDF4 import Dataset as NC

from cdo import Cdo
cdo = Cdo()

import numpy as np
import pandas as pa
import pylab as plt
import ogr

try:
    from pypismtools import unit_converter, smooth
except:
    from pypismtools.pypismtools import unit_converter, smooth

basin_list = ['CW', 'NE', 'NO', 'NW', 'SE', 'SW', 'GRIS']
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
parser.add_argument("-t", "--twinx", dest="twinx", action="store_true",
                    help='''adds a second ordinate with units mmSLE,
                  Default=False''', default=False)
parser.add_argument("--plot", dest="plot",
                    help='''What to plot.''',
                    choices=['anim_rcp_mass',
                             'basin_discharge', 'basin_smb', 'rel_basin_discharge', 'basin_mass', 'basin_mass_d',
                             'basin_d_cumulative',
                             'basin_rel_discharge',
                             'ens_mass',
                             'fluxes',
                             'flood_gate_length', 'flood_gate_area',
                             'per_basin_fluxes', 'per_basin_cumulative',
                             'rcp_mass', 'rcp_lapse_mass', 'rcp_d', 'rcp_flux', 'rcp_flux_rel',
                             'rcp_ens_mass',
                             'rcp_ens_d_flux', 'rcp_ens_smb_flux', 'rcp_ens_mass_flux',
                             'rcp_ens_area', 'rcp_ens_volume'],
                    default='basin_discharge')

parser.add_argument("--title", dest="title",
                    help='''Plot title.''', default=None)

options = parser.parse_args()
basin = options.basin
mrcp = options.mrcp
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
twinx = options.twinx
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


basin_col_dict = {'SW': '#542788',
                  'CW': '#b35806',
                  'NE': '#e08214',
                  'NO': '#fdb863',
                  'NW': '#b2abd2',
                  'SE': '#8073ac',
                  'GR': '#000000'}

rcp_col_dict = {'CTRL': 'k',
                '85': '#d94701',
                '45': '#fd8d3c',
                '26': '#fdbe85'}

rcp_dict = {'26': 'RCP 2.6',
            '45': 'RCP 4.5',
            '85': 'RCP 8.5',
            'CTRL': 'CTRL'}

flux_to_mass_vars_dict = {'tendency_of_ice_mass': 'ice_mass',
             'tendency_of_ice_mass_due_to_flow': 'flow_cumulative',
             'tendency_of_ice_mass_due_to_conservation_error': 'conservation_error_cumulative',
             'tendency_of_ice_mass_due_to_basal_mass_flux': 'basal_mass_flux_cumulative',
             'tendency_of_ice_mass_due_to_surface_mass_flux': 'surface_mass_flux_cumulative',
             'tendency_of_ice_mass_due_to_discharge': 'discharge_cumulative'}
flux_vars = flux_to_mass_vars_dict.keys()

flux_abbr_dict = {'tendency_of_ice_mass': '$\dot \mathregular{M}$',
                  'tendency_of_ice_mass_due_to_flow': 'divQ',
                  'tendency_of_ice_mass_due_to_conservation_error': '\dot e',
                  'tendency_of_ice_mass_due_to_basal_mass_flux': 'BMB',
                  'tendency_of_ice_mass_due_to_surface_mass_flux': 'SMB',
                  'tendency_of_ice_mass_due_to_discharge': 'D'}

flux_short_dict = {'tendency_of_ice_mass': 'dmdt',
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
             'tendency_of_ice_mass_due_to_discharge': '--'}

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

flux_plot_vars = ['tendency_of_ice_mass_due_to_discharge', 'tendency_of_ice_mass_due_to_surface_mass_flux']
mass_plot_vars = ['ice_mass']

flux_ounits = 'Gt year-1'
mass_ounits = 'Gt'

runmean_window = 11


def plot_flood_gate_area_ts():
    '''
    Plot time-series area of flood gates
    '''
    driver = ogr.GetDriverByName('ESRI Shapefile')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    for k, rcp in enumerate(rcp_list):
        ifile = ifiles[k]
        print('reading {}'.format(ifile))

        df = pa.read_csv(ifile)
        # Select marine areas (bed below sea level)
        ndf = df.loc[(df['bed'] < 0) & (df['thickness'] > 10)]

        dates = []
        fga = []
        nt = ndf['timestep'].max()
        for k in range(nt):
            fga.append(ndf.loc[ndf['timestep'] == k].sum()['bed'] * -500 / 2 /1e6)
            dates.append(k * 20)
        ax.plot(np.array(dates) + 2008, fga)

        # dates = np.array(dates)
        # t = np.array(t)
        # lengths = np.array(lengths)
        # ax.plot(t + start_year, lengths / 1e3,
        #         color=rcp_col_dict[rcp],
        #         lw=0.5,
        #         label=rcp_dict[rcp])

    # legend = ax.legend(loc="upper right",
    #                    edgecolor='0',
    #                    bbox_to_anchor=(0, 0, .35, 0.87),
    #                    bbox_transform=plt.gcf().transFigure)
    # legend.get_frame().set_linewidth(0.0)
    
    # ax.set_xlabel('Year (CE)')
    # ax.set_ylabel('length (km)')
        
    # if time_bounds:
    #     ax.set_xlim(time_bounds[0], time_bounds[1])

    # if bounds:
    #     ax.set_ylim(bounds[0], bounds[1])

    # ymin, ymax = ax.get_ylim()

    # if rotate_xticks:
    #     ticklabels = ax.get_xticklabels()
    #     for tick in ticklabels:
    #             tick.set_rotation(30)
    # else:
    #     ticklabels = ax.get_xticklabels()
    #     for tick in ticklabels:
    #         tick.set_rotation(0)
                    
    # if title is not None:
    #         plt.title(title)

    for out_format in out_formats:
        out_file = outfile  + '.' + out_format
        print "  - writing image %s ..." % out_file
        fig.savefig(out_file, bbox_inches='tight', dpi=out_res)

def plot_flood_gate_length_ts():
    '''
    Plot time-series length of flood gates
    '''
    driver = ogr.GetDriverByName('ESRI Shapefile')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    for k, rcp in enumerate(rcp_list):
        ifile = ifiles[k]
        print('reading {}'.format(ifile))
        print ifile
        ds = driver.Open(ifile)
        layer = ds.GetLayer()
        cnt = layer.GetFeatureCount()
        
        dates = []
        t = []
        lengths = []

        for feature in layer:
            dates.append(feature.GetField('timestamp'))
            t.append(feature.GetField('timestep'))
            geom = feature.GetGeometryRef()
            length = geom.GetArea() / 2.
            lengths.append(length)

        del ds

        dates = np.array(dates)
        t = np.array(t)
        lengths = np.array(lengths)
        ax.plot(t + start_year, lengths / 1e3,
                color=rcp_col_dict[rcp],
                lw=0.5,
                label=rcp_dict[rcp])

    if do_legend:
        legend = ax.legend(loc="upper right",
                           edgecolor='0',
                           bbox_to_anchor=(0, 0, .35, 0.87),
                           bbox_transform=plt.gcf().transFigure)
        legend.get_frame().set_linewidth(0.0)
    
    ax.set_xlabel('Year (CE)')
    ax.set_ylabel('length (km)')
        
    if time_bounds:
        ax.set_xlim(time_bounds[0], time_bounds[1])

    if bounds:
        ax.set_ylim(bounds[0], bounds[1])

    ymin, ymax = ax.get_ylim()

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
        out_file = outfile  + '.' + out_format
        print "  - writing image %s ..." % out_file
        fig.savefig(out_file, bbox_inches='tight', dpi=out_res)

def plot_rcp_flux():
    
    fig = plt.figure()
    offset = transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)
    ax = fig.add_subplot(111)

    for k, rcp in enumerate(rcp_list):
        ifile = ifiles[k]
        print('reading {}'.format(ifile))
        nc = NC(ifile, 'r')
        t = nc.variables["time"][:]

        date = np.arange(start_year + step,
                         start_year + (len(t[:]) + 1) * step,
                         step) 

        d_var_vals_sum = 0
        for d_var in ('tendency_of_ice_mass_due_to_discharge', 'tendency_of_ice_mass_due_to_basal_mass_flux'):
            d_var_vals = -np.squeeze(nc.variables[d_var][:])
            iunits = nc.variables[d_var].units
            d_var_vals_sum += unit_converter(d_var_vals, iunits, flux_ounits)

        nc.close()

        if runmean is not None:
            runmean_var_vals = smooth(d_var_vals_sum, window_len=runmean)
            plt.plot(date[:], d_var_vals_sum[:],
                     color=rcp_col_dict[rcp],
                     lw=0.25,
                     label=rcp)
            plt.plot(date[:], runmean_var_vals[:],
                     color=rcp_col_dict[rcp],
                     lw=0.5)
        else:
            plt.plot(date[:], d_var_vals_sum[:],
                     color=rcp_col_dict[rcp],
                     lw=0.5,
                     label=rcp)

    if do_legend:
        legend = ax.legend(loc="upper right",
                           edgecolor='0',
                           bbox_to_anchor=(0, 0, .35, 0.87),
                           bbox_transform=plt.gcf().transFigure)
        legend.get_frame().set_linewidth(0.0)
    
    ax.set_xlabel('Year (CE)')
    ax.set_ylabel('flux (Gt yr$^{-1}$)')
        
    if time_bounds:
        ax.set_xlim(time_bounds[0], time_bounds[1])

    if bounds:
        ax.set_ylim(bounds[0], bounds[1])

    ymin, ymax = ax.get_ylim()

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
        out_file = outfile + '_' + basin + '.' + out_format
        print "  - writing image %s ..." % out_file
        fig.savefig(out_file, bbox_inches='tight', dpi=out_res)

        
def plot_rcp_flux_relative():
    
    fig = plt.figure()
    offset = transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)
    ax = fig.add_subplot(111)

    for k, rcp in enumerate(rcp_list):
        ifile = ifiles[k]
        print('reading {}'.format(ifile))
        nc = NC(ifile, 'r')
        t = nc.variables["time"][:]

        date = np.arange(start_year + step,
                         start_year + (len(t[:]) + 1) * step,
                         step) 

        mass_var = 'tendency_of_ice_mass'
        mass_var_vals = -np.squeeze(nc.variables[mass_var][:])
        iunits = nc.variables[mass_var].units
        mass_var_vals = unit_converter(mass_var_vals, iunits, flux_ounits)

        
        d_var_vals_sum = 0
        for d_var in ('tendency_of_ice_mass_due_to_discharge', 'tendency_of_ice_mass_due_to_basal_mass_flux'):
            d_var_vals = -np.squeeze(nc.variables[d_var][:])
            iunits = nc.variables[d_var].units
            d_var_vals_sum += unit_converter(d_var_vals, iunits, flux_ounits)

        nc.close()

        runmean = 10
        runmean_d_var_vals = smooth(d_var_vals_sum, window_len=runmean)
        runmean_mass_var_vals = smooth(mass_var_vals, window_len=runmean)
        runmean_rel_flux_vals = runmean_d_var_vals / runmean_mass_var_vals * 100
        plt.plot(date[:], runmean_rel_flux_vals[:],
                 color=rcp_col_dict[rcp],
                 lw=0.5)

    if do_legend:
        legend = ax.legend(loc="upper right",
                           edgecolor='0',
                           bbox_to_anchor=(0, 0, .35, 0.87),
                           bbox_transform=plt.gcf().transFigure)
        legend.get_frame().set_linewidth(0.0)
    
    ax.set_xlabel('Year (CE)')
    ax.set_ylabel('rel. discharge (%)')
        
    if time_bounds:
        ax.set_xlim(time_bounds[0], time_bounds[1])

    if bounds:
        ax.set_ylim(bounds[0], bounds[1])

    ymin, ymax = ax.get_ylim()

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
        out_file = outfile + '_rel_' + basin + '.' + out_format
        print "  - writing image %s ..." % out_file
        fig.savefig(out_file, bbox_inches='tight', dpi=out_res)

    
def plot_fluxes(plot_vars=['discharge_flux']):
    
    fig = plt.figure()
    offset = transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)
    ax = fig.add_subplot(111)

    mcolors = ['#e41a1c',
               '#377eb8',
               '#4daf4a',
               '#984ea3',
               '#ff7f00']

    for ifile in ifiles:
        nc = NC(ifile, 'r')
        t = nc.variables["time"][:]

        date = np.arange(start_year + step,
                         start_year + (len(t[:]) + 1) * step,
                         step)
        for k, mvar in enumerate(plot_vars):
            var_vals = np.squeeze(nc.variables[mvar][:])
            iunits = nc.variables[mvar].units
            
            var_vals = unit_converter(var_vals, iunits, flux_ounits)
            runmean_var_vals = smooth(var_vals, window_len=runmean)
            plt.plot(date[:], runmean_var_vals[:],
                     color=mcolors[k],
                     lw=0.5,
                     label=mvar)
        nc.close()

    if do_legend:
        legend = ax.legend(loc="upper right",
                           edgecolor='0',
                           bbox_to_anchor=(0, 0, 1.25, 1),
                           bbox_transform=plt.gcf().transFigure)
        legend.get_frame().set_linewidth(0.2)

    if twinx:
        axSLE = ax.twinx()
        ax.set_autoscalex_on(False)
        axSLE.set_autoscalex_on(False)
        
    ax.set_xlabel('Year (CE)')
    ax.set_ylabel('mass flux (Gt yr$^{\mathregular{-1}}$)')
            
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


def plot_rcp_cold(plot_var='rel_area_cold'):
    
    fig = plt.figure()
    offset = transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)
    ax = fig.add_subplot(111)

    for k, rcp in enumerate(rcp_list):

        rcp_files = [f for f in ifiles if 'rcp_{}'.format(rcp) in f]

        print('Reading files for {}'.format(rcp_dict[rcp]))

        cdf_ensmin = cdo.ensmin(input=rcp_files, returnCdf=True)
        cdf_ensmax = cdo.ensmax(input=rcp_files, returnCdf=True)
        cdf_ensmean = cdo.ensmean(input=rcp_files, returnCdf=True)
        t = cdf_ensmax.variables['time'][:]

        ensmin_vals = cdf_ensmin.variables[plot_var][:] * 100
        ensmax_vals = cdf_ensmax.variables[plot_var][:] * 100
        ensmean_vals = cdf_ensmean.variables[plot_var][:] * 100

        date = np.arange(start_year + step,
                         start_year + (len(t[:]) + 1) * step,
                         step) 

        ax.fill_between(date[:], ensmin_vals, ensmax_vals,
                        color=rcp_col_dict[rcp],
                        linewidth=0,
                        label=rcp_dict[rcp])


    if do_legend:
        legend = ax.legend(loc="upper right",
                           edgecolor='0',
                           bbox_to_anchor=(0, 0, .35, 0.87),
                           bbox_transform=plt.gcf().transFigure)
        legend.get_frame().set_linewidth(0.0)
    
    ax.set_xlabel('Year (CE)')
    ax.set_ylabel('(%)')
        
    if time_bounds:
        ax.set_xlim(time_bounds[0], time_bounds[1])

    if bounds:
        ax.set_ylim(bounds[0], bounds[1])

    ymin, ymax = ax.get_ylim()

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
        print "  - writing image %s ..." % out_file
        fig.savefig(out_file, bbox_inches='tight', dpi=out_res)

def plot_rcp_ens_mass(plot_var=mass_plot_vars):
    
    fig = plt.figure()
    offset = transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)
    ax = fig.add_subplot(111)
    
    for k, rcp in enumerate(rcp_list):

        rcp_files = [f for f in ifiles if 'rcp_{}'.format(rcp) in f]

        print('Reading files for {}'.format(rcp_dict[rcp]))
        
        cdf_mass_ensstd = cdo.ensstd(input=rcp_files, returnCdf=True, options=pthreads)
        cdf_mass_ensmean = cdo.ensmean(input=rcp_files, returnCdf=True, options=pthreads)
        t = cdf_mass_ensmean.variables['time'][:]

        mass_ensstd_vals = cdf_mass_ensstd.variables[plot_var][:] - cdf_mass_ensstd.variables[plot_var][0]
        iunits = cdf_mass_ensstd[plot_var].units
        mass_ensstd_vals = -unit_converter(mass_ensstd_vals, iunits, mass_ounits) * gt2mSLE

        mass_ensmean_vals = cdf_mass_ensmean.variables[plot_var][:] - cdf_mass_ensmean.variables[plot_var][0]
        iunits = cdf_mass_ensmean[plot_var].units
        mass_ensmean_vals = -unit_converter(mass_ensmean_vals, iunits, mass_ounits) * gt2mSLE

        date = np.arange(start_year + step,
                         start_year + (len(t[:]) + 1) * step,
                         step) 


        # ensemble +- 1 sigma
        ax.fill_between(date[:], mass_ensmean_vals-mass_ensstd_vals, mass_ensmean_vals+mass_ensstd_vals,
                        color=rcp_col_dict[rcp],
                        alpha=0.5,
                        linewidth=0)
        
        ax.plot(date[:], mass_ensmean_vals,
                        color=rcp_col_dict[rcp],
                        linewidth=0.5,
                        label=rcp_dict[rcp])

        ax.plot(date[:], mass_ensmean_vals+mass_ensstd_vals,
                color=rcp_col_dict[rcp],
                linestyle='dashed',
                linewidth=0.25)

        ax.plot(date[:], mass_ensmean_vals-mass_ensstd_vals,
                color=rcp_col_dict[rcp],
                linestyle='dashed',
                linewidth=0.25)

        idx = np.where(np.array(date) == time_bounds[-1])[0][0]
        m_mean = mass_ensmean_vals[idx]
        m_std = np.abs(mass_ensstd_vals[idx])
        
        x_sle, y_sle = time_bounds[-1], m_mean
        plt.text( x_sle, y_sle, '{: 1.2f}$\pm${:1.2f}'.format(m_mean, m_std),
                  color=rcp_col_dict[rcp])

    if do_legend:
        legend = ax.legend(loc="upper right",
                           edgecolor='0',
                           bbox_to_anchor=(0, 0, .35, 0.88),
                           bbox_transform=plt.gcf().transFigure)
        legend.get_frame().set_linewidth(0.0)
    
    ax.set_xlabel('Year (CE)')
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
        out_file = outfile + '_rcp' + '_'  + plot_var + '.' + out_format
        print "  - writing image %s ..." % out_file
        fig.savefig(out_file, bbox_inches='tight', dpi=out_res)


def plot_ens_mass(plot_var=mass_plot_vars):
    
    fig = plt.figure()
    offset = transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)
    ax = fig.add_subplot(111)
    
    rcp_files = ifiles
    rcp = '45'

    print('Reading files for {}'.format(rcp_dict[rcp]))

    cdf_mass_ensmin = cdo.ensmin(input=rcp_files, returnCdf=True)
    cdf_mass_ensmax = cdo.ensmax(input=rcp_files, returnCdf=True)
    cdf_mass_ensstd = cdo.ensstd(input=rcp_files, returnCdf=True)
    cdf_mass_ensmean = cdo.ensmean(input=rcp_files, returnCdf=True)
    t = cdf_mass_ensmax.variables['time'][:]

    mass_ensmin_vals = cdf_mass_ensmin.variables[plot_var][:] - cdf_mass_ensmin.variables[plot_var][0]
    iunits = cdf_mass_ensmin[plot_var].units
    mass_ensmin_vals = -unit_converter(mass_ensmin_vals, iunits, mass_ounits) * gt2mSLE

    mass_ensmax_vals = cdf_mass_ensmax.variables[plot_var][:] -  cdf_mass_ensmax.variables[plot_var][0]
    iunits = cdf_mass_ensmax[plot_var].units
    mass_ensmax_vals = -unit_converter(mass_ensmax_vals, iunits, mass_ounits) * gt2mSLE

    mass_ensstd_vals = cdf_mass_ensstd.variables[plot_var][:] - cdf_mass_ensstd.variables[plot_var][0]
    iunits = cdf_mass_ensstd[plot_var].units
    mass_ensstd_vals = -unit_converter(mass_ensstd_vals, iunits, mass_ounits) * gt2mSLE

    mass_ensmean_vals = cdf_mass_ensmean.variables[plot_var][:] - cdf_mass_ensmean.variables[plot_var][0]
    iunits = cdf_mass_ensmean[plot_var].units
    mass_ensmean_vals = -unit_converter(mass_ensmean_vals, iunits, mass_ounits) * gt2mSLE

    date = np.arange(start_year + step,
                     start_year + (len(t[:]) + 1) * step,
                     step) 

    # ensemble min/max
    ax.fill_between(date[:], mass_ensmin_vals, mass_ensmax_vals,
                    alpha=0.5,
                    color=rcp_col_dict[rcp],
                    linewidth=0)

    # ensemble +- 1 sigma
    ax.fill_between(date[:], mass_ensmean_vals-mass_ensstd_vals, mass_ensmean_vals+mass_ensstd_vals,
                    color=rcp_col_dict[rcp],
                    linewidth=0)

    ax.plot(date[:], mass_ensmean_vals,
                    color='k',
                    linewidth=0.5)

    idx = np.where(np.array(date) == time_bounds[-1])[0][0]
    m_max = mass_ensmax_vals[idx]
    m_min = mass_ensmin_vals[idx]
    m_mean = mass_ensmean_vals[idx]
    m_std = mass_ensstd_vals[idx]
    m_rel = np.abs(m_max - m_min)

    print('MASS dGMSL {}: {:1.2f} - {:1.2f}, mean {:1.2f}; DIFF {:1.2f}'.format(time_bounds[-1],  m_max, m_min, m_mean, m_rel))

    x_sle, y_sle = time_bounds[-1], m_mean
    plt.text( x_sle, y_sle, '{: 1.2f}$\pm${: 1.2f}'.format(m_mean, m_std),
              color=rcp_col_dict[rcp])

    # if do_legend:
    #     legend = ax.legend(loc="upper right",
    #                        edgecolor='0',
    #                        bbox_to_anchor=(0, 0, .35, 0.88),
    #                        bbox_transform=plt.gcf().transFigure)
    #     legend.get_frame().set_linewidth(0.0)
    
    ax.set_xlabel('Year (CE)')
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
        out_file = outfile + '_rcp' + '_'  + plot_var + '.' + out_format
        print "  - writing image %s ..." % out_file
        fig.savefig(out_file, bbox_inches='tight', dpi=out_res)


def anim_rcp_mass(plot_var=mass_plot_vars):

    for frame in range(1000):
    
        fig = plt.figure()
        offset = transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)
        ax = fig.add_subplot(111)
        ax.grid(axis='y', lw=0.15, color='k', ls=':')
        
        for k, rcp in enumerate(rcp_list):

            rcp_files = [f for f in ifiles if 'rcp_{}'.format(rcp) in f]

            print('Reading files for {}'.format(rcp_dict[rcp]))

            cdf_mass_ensmean = cdo.ensmean(input=rcp_files, returnCdf=True)

            mass_ensmean_vals = cdf_mass_ensmean.variables[plot_var][:]
            iunits = cdf_mass_ensmean[plot_var].units
            mass_ensmean_vals = -unit_converter(mass_ensmean_vals, iunits, mass_ounits) * gt2mSLE

            t = cdf_mass_ensmean.variables['time'][:]

            date = np.arange(start_year + step,
                             start_year + (len(t[:]) + 1) * step,
                             step) 


            ax.plot(date[:], mass_ensmean_vals,
                    alpha=0.5,
                    color=rcp_col_dict[rcp],
                    linewidth=0.5)

            ax.plot(date[0:frame], mass_ensmean_vals[0:frame],
                    color=rcp_col_dict[rcp],
                    linewidth=0.5)
                
            ax.plot(date[frame], mass_ensmean_vals[frame],
                    marker='o',
                    markersize=2,
                    color=rcp_col_dict[rcp],
                    linewidth=0,
                    label=rcp_dict[rcp])


        if do_legend:
            legend = ax.legend(loc="upper right",
                               edgecolor='0',
                               bbox_to_anchor=(0, 0, .35, 0.88),
                               bbox_transform=plt.gcf().transFigure)
            legend.get_frame().set_linewidth(0.0)

        ax.set_xlabel('Year (CE)')
        ax.set_ylabel('sea-level (m)')

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
            out_file = '{}_{}_{:04}.{}'.format(outfile, plot_var, frame, out_format)
            print "  - writing image %s ..." % out_file
            fig.savefig(out_file, bbox_inches='tight', dpi=out_res)


def plot_rcp_ens_flux(plot_var=mass_plot_vars):
    
    fig = plt.figure()
    offset = transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)
    ax = fig.add_subplot(111)

    for k, rcp in enumerate(rcp_list):

        rcp_files = [f for f in ifiles if 'rcp_{}'.format(rcp) in f]

        print('Reading files for {}'.format(rcp_dict[rcp]))

        if openmp_n > 1:
            tmp_flux_ensmin = cdo.ensmin(input=rcp_files, options='-P {N}'.format(N=openmp_n))
            tmp_flux_ensmax = cdo.ensmax(input=rcp_files, options='-P {N}'.format(N=openmp_n))
            tmp_flux_ensmean = cdo.ensmean(input=rcp_files, options='-P {N}'.format(N=openmp_n))
        else:
            tmp_flux_ensmin = cdo.ensmin(input=rcp_files)
            tmp_flux_ensmax = cdo.ensmax(input=rcp_files)
            tmp_flux_ensmean = cdo.ensmean(input=rcp_files)

        cdf_flux_ensmin = cdo.runmean(runmean_window, input=tmp_flux_ensmin, returnCdf=True)
        cdf_flux_ensmax = cdo.runmean(runmean_window, input=tmp_flux_ensmax, returnCdf=True)
        cdf_flux_ensmean = cdo.runmean(runmean_window, input=tmp_flux_ensmean, returnCdf=True)

        t = cdf_flux_ensmax.variables['time'][:]

        mass_ensmin = cdf_flux_ensmin.variables['mass_glacierized']
        iunits = mass_ensmin.units
        mass_ensmin_vals = unit_converter(mass_ensmin[:], iunits, mass_ounits)

        mass_ensmax = cdf_flux_ensmax.variables['mass_glacierized']
        iunits = mass_ensmax.units
        mass_ensmax_vals = unit_converter(mass_ensmax[:], iunits, mass_ounits)

        mass_ensmean = cdf_flux_ensmean.variables['mass_glacierized']
        iunits = mass_ensmean.units
        mass_ensmean_vals = unit_converter(mass_ensmean[:], iunits, mass_ounits)

        mass_flux_ensmin = cdf_flux_ensmin.variables['mass_rate_of_change_glacierized']
        iunits = mass_flux_ensmin.units
        mass_flux_ensmin_vals = -unit_converter(mass_flux_ensmin[:], iunits, flux_ounits)

        mass_flux_ensmax = cdf_flux_ensmax.variables['mass_rate_of_change_glacierized']
        iunits = mass_flux_ensmax.units
        mass_flux_ensmax_vals = -unit_converter(mass_flux_ensmax[:], iunits, flux_ounits) 

        mass_flux_ensmean = cdf_flux_ensmean.variables['mass_rate_of_change_glacierized']
        iunits = mass_flux_ensmean.units
        mass_flux_ensmean_vals = -unit_converter(mass_flux_ensmean[:], iunits, flux_ounits)

        smb_flux_ensmin = cdf_flux_ensmin.variables['surface_ice_flux']
        iunits = smb_flux_ensmin.units
        smb_flux_ensmin_vals = -unit_converter(smb_flux_ensmin[:], iunits, flux_ounits)

        smb_flux_ensmax = cdf_flux_ensmax.variables['surface_ice_flux']
        iunits = smb_flux_ensmax.units
        smb_flux_ensmax_vals = -unit_converter(smb_flux_ensmax[:], iunits, flux_ounits) 

        smb_flux_ensmean = cdf_flux_ensmean.variables['surface_ice_flux']
        iunits = smb_flux_ensmean.units
        smb_flux_ensmean_vals = -unit_converter(smb_flux_ensmean[:], iunits, flux_ounits)

        if openmp_n > 1:
            tmp_ensmin = cdo.ensmin(input=rcp_files, options='-P {N}'.format(N=openmp_n))
            tmp_ensmax = cdo.ensmax(input=rcp_files, options='-P {N}'.format(N=openmp_n))
            tmp_ensmean = cdo.ensmean(input=rcp_files, options='-P {N}'.format(N=openmp_n))
            tmp_ensstd = cdo.ensstd(input=rcp_files, options='-P {N}'.format(N=openmp_n))
        else:
            tmp_ensmin = cdo.ensmin(input=rcp_files)
            tmp_ensmax = cdo.ensmax(input=rcp_files)
            tmp_ensmean = cdo.ensmean(input=rcp_files)
            tmp_ensstd = cdo.ensstd(input=rcp_files)

        cdf_d_ensmin = cdo.runmean(runmean_window, input='-expr,discharge=discharge_flux+sub_shelf_ice_flux+grounded_basal_ice_flux {}'.format(tmp_ensmin), returnCdf=True)
        cdf_d_ensmax = cdo.runmean(runmean_window, input='-expr,discharge=discharge_flux+sub_shelf_ice_flux+grounded_basal_ice_flux {}'.format(tmp_ensmax), returnCdf=True)
        cdf_d_ensmean = cdo.runmean(runmean_window, input='-expr,discharge=discharge_flux+sub_shelf_ice_flux+grounded_basal_ice_flux {}'.format(tmp_ensmean), returnCdf=True)
        cdf_d_ensstd = cdo.runmean(runmean_window, input='-expr,discharge=discharge_flux+sub_shelf_ice_flux+grounded_basal_ice_flux {}'.format(tmp_ensstd), returnCdf=True)

        d_flux_ensmin = cdf_d_ensmin.variables['discharge']
        d_flux_ensmin_vals = -unit_converter(d_flux_ensmin[:], iunits, flux_ounits)

        d_flux_ensmax = cdf_d_ensmax.variables['discharge']
        d_flux_ensmax_vals = -unit_converter(d_flux_ensmax[:], iunits, flux_ounits)

        d_flux_ensmean = cdf_d_ensmean.variables['discharge']
        d_flux_ensmean_vals = -unit_converter(d_flux_ensmean[:], iunits, flux_ounits)

        d_flux_ensstd = cdf_d_ensstd.variables['discharge']
        d_flux_ensstd_vals = -unit_converter(d_flux_ensstd[:], iunits, flux_ounits)

        date = np.arange(start_year + step,
                         start_year + (len(t[:]) + 1) * step,
                         step) 


        if plot_var == 'discharge':
            ensmean_vals = d_flux_ensmean_vals
            ensmin_vals = d_flux_ensmin_vals
            ensmax_vals = d_flux_ensmax_vals
        elif plot_var == 'smb':
            ensmean_vals = smb_flux_ensmean_vals
            ensmin_vals = smb_flux_ensmin_vals
            ensmax_vals = smb_flux_ensmax_vals
        else:
            ensmean_vals = mass_flux_ensmean_vals
            ensmin_vals = mass_flux_ensmin_vals
            ensmax_vals = mass_flux_ensmax_vals

        rel = False
        if rel:
            ensmean_vals /= mass_flux_ensmean_vals
            ensmin_vals /= mass_flux_ensmin_vals
            ensmax_vals /= mass_flux_ensmax_vals
            ensmean_vals *= 100
            ensmin_vals *= 100
            ensmax_vals *= 100
            
        ax.fill_between(date[:], ensmin_vals * gt2cmSLE, ensmax_vals * gt2cmSLE,
                        color=rcp_col_dict[rcp],
                        linewidth=0,
                        label=rcp_dict[rcp])

        # ensemble mean
        ax.plot(date[:],  ensmean_vals * gt2cmSLE,
                color='w',
                linestyle=':',
                linewidth=0.2)

    if do_legend:
        legend = ax.legend(loc="upper right",
                           edgecolor='0',
                           bbox_to_anchor=(0, 0, .35, 0.28),
                           bbox_transform=plt.gcf().transFigure)
        legend.get_frame().set_linewidth(0.0)

    # 2014 rates 
    p_sle_2014 = ax.fill_between([2010, 2020], 0.29 - 0.03, 0.29 + 0.03,
                                color='0.8',
                                linewidth=0)
    p_sle_2014.set_zorder(-101)
    l_sle_2014 = ax.hlines(0.29, 2010, 2020,
                           linewidth=0.2, linestyle=':', color='0.6')
    l_sle_2014.set_zorder(-100)
    
    if rel:
        ax.hlines(25, time_bounds[0], time_bounds[1], linewidth=0.2, linestyle=':')
    ax.set_xlabel('Year (CE)')
    if rel:
        ax.set_ylabel('rel. contribution (%)')
    else:
        ax.set_ylabel('sea-level rise (mm yr$^{-1}$)')
        
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
        out_file = outfile + '_rcp' + '_'  + plot_var + '_flux.' + out_format
        print "  - writing image %s ..." % out_file
        fig.savefig(out_file, bbox_inches='tight', dpi=out_res)


def plot_ens_flux(plot_var=mass_plot_vars):
    
    for k, rcp in enumerate(rcp_list):

        rcp_files = [f for f in ifiles if 'rcp_{}'.format(rcp) in f]

        fig = plt.figure()
        offset = transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)
        ax = fig.add_subplot(111)

        print('Reading files for {}'.format(rcp_dict[rcp]))

        if openmp_n > 1:
            tmp_flux_ensmin = cdo.ensmin(input=rcp_files, options='-P {N}'.format(N=openmp_n))
            tmp_flux_ensmax = cdo.ensmax(input=rcp_files, options='-P {N}'.format(N=openmp_n))
            tmp_flux_ensmean = cdo.ensmean(input=rcp_files, options='-P {N}'.format(N=openmp_n))
        else:
            tmp_flux_ensmin = cdo.ensmin(input=rcp_files)
            tmp_flux_ensmax = cdo.ensmax(input=rcp_files)
            tmp_flux_ensmean = cdo.ensmean(input=rcp_files)

        cdf_flux_ensmin = cdo.runmean(runmean_window, input=tmp_flux_ensmin, returnCdf=True)
        cdf_flux_ensmax = cdo.runmean(runmean_window, input=tmp_flux_ensmax, returnCdf=True)
        cdf_flux_ensmean = cdo.runmean(runmean_window, input=tmp_flux_ensmean, returnCdf=True)

        t = cdf_flux_ensmax.variables['time'][:]

        mass_ensmin = cdf_flux_ensmin.variables['mass_glacierized']
        iunits = mass_ensmin.units
        mass_ensmin_vals = unit_converter(mass_ensmin[:], iunits, mass_ounits)

        mass_ensmax = cdf_flux_ensmax.variables['mass_glacierized']
        iunits = mass_ensmax.units
        mass_ensmax_vals = unit_converter(mass_ensmax[:], iunits, mass_ounits)

        mass_ensmean = cdf_flux_ensmean.variables['mass_glacierized']
        iunits = mass_ensmean.units
        mass_ensmean_vals = unit_converter(mass_ensmean[:], iunits, mass_ounits)

        mass_flux_ensmin = cdf_flux_ensmin.variables['mass_rate_of_change_glacierized']
        iunits = mass_flux_ensmin.units
        mass_flux_ensmin_vals = -unit_converter(mass_flux_ensmin[:], iunits, flux_ounits)

        mass_flux_ensmax = cdf_flux_ensmax.variables['mass_rate_of_change_glacierized']
        iunits = mass_flux_ensmax.units
        mass_flux_ensmax_vals = -unit_converter(mass_flux_ensmax[:], iunits, flux_ounits) 

        mass_flux_ensmean = cdf_flux_ensmean.variables['mass_rate_of_change_glacierized']
        iunits = mass_flux_ensmean.units
        mass_flux_ensmean_vals = -unit_converter(mass_flux_ensmean[:], iunits, flux_ounits)

        smb_flux_ensmin = cdf_flux_ensmin.variables['surface_ice_flux']
        iunits = smb_flux_ensmin.units
        smb_flux_ensmin_vals = -unit_converter(smb_flux_ensmin[:], iunits, flux_ounits)

        smb_flux_ensmax = cdf_flux_ensmax.variables['surface_ice_flux']
        iunits = smb_flux_ensmax.units
        smb_flux_ensmax_vals = -unit_converter(smb_flux_ensmax[:], iunits, flux_ounits) 

        smb_flux_ensmean = cdf_flux_ensmean.variables['surface_ice_flux']
        iunits = smb_flux_ensmean.units
        smb_flux_ensmean_vals = -unit_converter(smb_flux_ensmean[:], iunits, flux_ounits)

        if openmp_n > 1:
            tmp_ensmin = cdo.ensmin(input=rcp_files, options='-P {N}'.format(N=openmp_n))
            tmp_ensmax = cdo.ensmax(input=rcp_files, options='-P {N}'.format(N=openmp_n))
            tmp_ensmean = cdo.ensmean(input=rcp_files, options='-P {N}'.format(N=openmp_n))
            tmp_ensstd = cdo.ensstd(input=rcp_files, options='-P {N}'.format(N=openmp_n))
        else:
            tmp_ensmin = cdo.ensmin(input=rcp_files)
            tmp_ensmax = cdo.ensmax(input=rcp_files)
            tmp_ensmean = cdo.ensmean(input=rcp_files)
            tmp_ensstd = cdo.ensstd(input=rcp_files)

        cdf_d_ensmin = cdo.runmean(runmean_window, input='-expr,discharge=discharge_flux+sub_shelf_ice_flux+grounded_basal_ice_flux {}'.format(tmp_ensmin), returnCdf=True)
        cdf_d_ensmax = cdo.runmean(runmean_window, input='-expr,discharge=discharge_flux+sub_shelf_ice_flux+grounded_basal_ice_flux {}'.format(tmp_ensmax), returnCdf=True)
        cdf_d_ensmean = cdo.runmean(runmean_window, input='-expr,discharge=discharge_flux+sub_shelf_ice_flux+grounded_basal_ice_flux {}'.format(tmp_ensmean), returnCdf=True)
        cdf_d_ensstd = cdo.runmean(runmean_window, input='-expr,discharge=discharge_flux+sub_shelf_ice_flux+grounded_basal_ice_flux {}'.format(tmp_ensstd), returnCdf=True)

        d_flux_ensmin = cdf_d_ensmin.variables['discharge']
        d_flux_ensmin_vals = -unit_converter(d_flux_ensmin[:], iunits, flux_ounits)

        d_flux_ensmax = cdf_d_ensmax.variables['discharge']
        d_flux_ensmax_vals = -unit_converter(d_flux_ensmax[:], iunits, flux_ounits)

        d_flux_ensmean = cdf_d_ensmean.variables['discharge']
        d_flux_ensmean_vals = -unit_converter(d_flux_ensmean[:], iunits, flux_ounits)

        d_flux_ensstd = cdf_d_ensstd.variables['discharge']
        d_flux_ensstd_vals = -unit_converter(d_flux_ensstd[:], iunits, flux_ounits)

        date = np.arange(start_year + step,
                         start_year + (len(t[:]) + 1) * step,
                         step) 


        if plot_var == 'discharge':
            ensmean_vals = d_flux_ensmean_vals
            ensmin_vals = d_flux_ensmin_vals
            ensmax_vals = d_flux_ensmax_vals
        elif plot_var == 'smb':
            ensmean_vals = smb_flux_ensmean_vals
            ensmin_vals = smb_flux_ensmin_vals
            ensmax_vals = smb_flux_ensmax_vals
        else:
            ensmean_vals = mass_flux_ensmean_vals
            ensmin_vals = mass_flux_ensmin_vals
            ensmax_vals = mass_flux_ensmax_vals

        rel = False
        if rel:
            ensmean_vals /= mass_flux_ensmean_vals
            ensmin_vals /= mass_flux_ensmin_vals
            ensmax_vals /= mass_flux_ensmax_vals
            ensmean_vals *= 100
            ensmin_vals *= 100
            ensmax_vals *= 100
            
        ax.fill_between(date[:], ensmin_vals * gt2cmSLE, ensmax_vals * gt2cmSLE,
                        color=rcp_col_dict[rcp],
                        linewidth=0,
                        label=rcp_dict[rcp])

        # ensemble mean
        ax.plot(date[:],  ensmean_vals * gt2cmSLE,
                color='w',
                linestyle=':',
                linewidth=0.2)

    if do_legend:
        legend = ax.legend(loc="upper right",
                           edgecolor='0',
                           bbox_to_anchor=(0, 0, .35, 0.28),
                           bbox_transform=plt.gcf().transFigure)
        legend.get_frame().set_linewidth(0.0)

    # 2014 rates 
    p_sle_2014 = ax.fill_between([2010, 2020], 0.29 - 0.03, 0.29 + 0.03,
                                color='0.8',
                                linewidth=0)
    p_sle_2014.set_zorder(-101)
    l_sle_2014 = ax.hlines(0.29, 2010, 2020,
                           linewidth=0.2, linestyle=':', color='0.6')
    l_sle_2014.set_zorder(-100)
    
    if rel:
        ax.hlines(25, time_bounds[0], time_bounds[1], linewidth=0.2, linestyle=':')
    ax.set_xlabel('Year (CE)')
    if rel:
        ax.set_ylabel('rel. contribution (%)')
    else:
        ax.set_ylabel('sea-level rise (mm yr$^{-1}$)')
        
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
        out_file = outfile + '_rcp' + '_'  + plot_var + '_flux.' + out_format
        print "  - writing image %s ..." % out_file
        fig.savefig(out_file, bbox_inches='tight', dpi=out_res)
        
def plot_rcp_mass(plot_var=mass_plot_vars):
    
    fig = plt.figure()
    offset = transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)
    ax = fig.add_subplot(111)

    for k, rcp in enumerate(rcp_list):
        ifile = ifiles[k]
        print('reading {}'.format(ifile))
        nc = NC(ifile, 'r')
        t = nc.variables["time"][:]

        date = np.arange(start_year + step,
                         start_year + (len(t[:]) + 1) * step,
                         step) 
        mvar = plot_var
        var_vals = np.squeeze(nc.variables[mvar][:])
        iunits = nc.variables[mvar].units
        var_vals = -unit_converter(var_vals, iunits, mass_ounits) * gt2mSLE
        plt.plot(date[:], var_vals[:],
                 color=rcp_col_dict[rcp],
                 lw=0.5,
                 label=rcp_dict[rcp])

        d_var_vals_sum = 0
        for d_var in ('discharge_cumulative', 'sub_shelf_ice_flux_cumulative', 'basal_ice_flux_cumulative', 'grounded_basal_ice_flux_cumulative'):
            try:
                d_var_vals = -np.squeeze(nc.variables[d_var][:]) * gt2mSLE
                iunits = nc.variables[d_var].units
                d_var_vals_sum += unit_converter(d_var_vals, iunits, mass_ounits)
            except:
                pass

        nc.close()

        idx = np.where(np.array(date) == time_bounds[-1])[0][0]

        x_sle, y_sle = time_bounds[-1], var_vals[idx]
        d_percent = d_var_vals_sum[idx] / var_vals[idx] * 100
        plt.text( x_sle, y_sle, '{: 3.0f}%'.format(d_percent),
                  color=rcp_col_dict[rcp])

        
        idx_2100 = np.where(np.array(date) == 2100)
        idx_2200 = np.where(np.array(date) == 2200)
        idx_3000 = np.where(np.array(date) == 3000)

        print('''RCP {}:
          Year 2100: {: 1.2f}m
          Year 2200: {: 1.2f}m
          Year 3000: {: 1.2f}m\n'''.format(rcp, var_vals[idx_2100][0], var_vals[idx_2200][0], var_vals[idx_3000][0]))

    if do_legend:
        legend = ax.legend(loc="upper right",
                           edgecolor='0',
                           bbox_to_anchor=(0, 0, .35, 0.87),
                           bbox_transform=plt.gcf().transFigure)
        legend.get_frame().set_linewidth(0.0)
    
    ax.set_xlabel('Year (CE)')
    ax.set_ylabel('$\Delta$(GMSL) (m)')
        
    if time_bounds:
        ax.set_xlim(time_bounds[0], time_bounds[1])

    if bounds:
        ax.set_ylim(bounds[0], bounds[1])

    ymin, ymax = ax.get_ylim()

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
        print "  - writing image %s ..." % out_file
        fig.savefig(out_file, bbox_inches='tight', dpi=out_res)


def plot_rcp_lapse_mass(plot_var=mass_plot_vars):
    
    fig = plt.figure()
    offset = transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)
    ax = fig.add_subplot(111)

    print ifiles

    sle_lapse_6 = []
    for k, rcp in enumerate(rcp_list):
        ifile = ifiles[k]
        print('reading {}'.format(ifile))
        nc = NC(ifile, 'r')
        t = nc.variables["time"][:]

        date = np.arange(start_year + step,
                         start_year + (len(t[:]) + 1) * step,
                         step) 
        mvar = plot_var
        var_vals = np.squeeze(nc.variables[mvar][:])
        iunits = nc.variables[mvar].units
        var_vals = -unit_converter(var_vals, iunits, mass_ounits) * gt2mSLE
        sle_lapse_6.append(var_vals[-1])
        plt.plot(date[:], var_vals[:],
                 color=rcp_col_dict[rcp],
                 lw=0.5,
                 label=rcp_dict[rcp])
        nc.close()
    m = k
    sle_lapse_0 = []
    for k, rcp in enumerate(rcp_list):
        ifile = ifiles[k+m+1]
        print('reading {}'.format(ifile))
        nc = NC(ifile, 'r')
        t = nc.variables["time"][:]

        date = np.arange(start_year + step,
                         start_year + (len(t[:]) + 1) * step,
                         step) 
        mvar = plot_var
        var_vals = np.squeeze(nc.variables[mvar][:])
        iunits = nc.variables[mvar].units
        var_vals = -unit_converter(var_vals, iunits, mass_ounits) * gt2mSLE
        sle_lapse_0.append(var_vals[-1])
        plt.plot(date[:], var_vals[:],
                 color=rcp_col_dict[rcp],
                 lw=0.5,
                 ls=':')
        nc.close()

    for k, rcp in enumerate(rcp_list):
        x_sle, y_sle = time_bounds[-1], sle_lapse_6[k]
        sle_percent_diff = (sle_lapse_6[k] - sle_lapse_0[k]) / sle_lapse_0[k] * 100
        plt.text( x_sle, y_sle, '{: 3.0f}%'.format(sle_percent_diff),
                          color=rcp_col_dict[rcp])

    if do_legend:
        legend = ax.legend(loc="upper right",
                           edgecolor='0',
                           bbox_to_anchor=(0, 0, .35, 0.87),
                           bbox_transform=plt.gcf().transFigure)
        legend.get_frame().set_linewidth(0.0)
    
    ax.set_xlabel('Year (CE)')
    ax.set_ylabel('$\Delta$(GMSL) (m)')
        
    if time_bounds:
        ax.set_xlim(time_bounds[0], time_bounds[1])

    if bounds:
        ax.set_ylim(bounds[0], bounds[1])

    ymin, ymax = ax.get_ylim()

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
        print "  - writing image %s ..." % out_file
        fig.savefig(out_file, bbox_inches='tight', dpi=out_res)


def plot_ens_d_by_basin():
    '''
    Make a plot per basin with all flux_plot_vars
    '''
    
    for basin in basins_list:

        fig = plt.figure()
        offset = transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)
        ax = fig.add_subplot(111)

        basin_files = [f for f in ifiles if 'b_{}'.format(basin) in f]

        print('Reading files for {}'.format(rcp_dict[rcp]))

        date = np.arange(start_year + step,
                         start_year + (len(t[:]) + 1) * step,
                         step) 

        if do_legend:
            legend = ax.legend(loc="upper right",
                               edgecolor='0',
                               bbox_to_anchor=(0, 0, 1.15, 1),
                               bbox_transform=plt.gcf().transFigure)
            legend.get_frame().set_linewidth(0.2)
        
        ax.set_xlabel('Year (CE)')
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
            out_file = outfile + '_basin_{}'.format(basin)  + '_fluxes.' + out_format
            print "  - writing image %s ..." % out_file
            fig.savefig(out_file, bbox_inches='tight', dpi=out_res)


def plot_fluxes_by_basin(plot_vars=['tendency_of_ice_mass', 'tendency_of_ice_mass_due_to_discharge', 'tendency_of_ice_mass_due_to_surface_mass_flux']):
    '''
    Make a plot per basin with all flux_plot_vars
    '''
    
    for k, ifile in enumerate(ifiles):

        fig = plt.figure()
        offset = transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)
        ax = fig.add_subplot(111)

        basin = basins_list[k]
        print('reading {}'.format(ifile))
        nc = NC(ifile, 'r')
        t = nc.variables["time"][:]

        date = np.arange(start_year + step,
                         start_year + (len(t[:]) + 1) * step,
                         step) 

        for mvar in plot_vars:
            var_vals = np.squeeze(nc.variables[mvar][:])
            iunits = nc.variables[mvar].units

            var_vals = -unit_converter(var_vals, iunits, flux_ounits) * gt2mSLE
            if runmean is not None:
                runmean_var_vals = smooth(var_vals, window_len=runmean)
                plt.plot(date[:], var_vals[:],
                         color=basin_col_dict[basin],
                         lw=0.25,
                         ls=flux_style_dict[mvar])
                plt.plot(date[:], runmean_var_vals[:],
                         color=basin_col_dict[basin],
                         lw=0.5,
                         ls=flux_style_dict[mvar],
                         label=flux_abbr_dict[mvar])
            else:
                plt.plot(date[:], var_vals[:],
                         color=basin_col_dict[basin],
                         lw=0.5,
                         ls=flux_style_dict[mvar],
                         label=flux_abbr_dict[mvar])
        nc.close()

        if do_legend:
            legend = ax.legend(loc="upper right",
                               edgecolor='0',
                               bbox_to_anchor=(0, 0, 1.15, 1),
                               bbox_transform=plt.gcf().transFigure)
            legend.get_frame().set_linewidth(0.2)
    
        ax.set_xlabel('Year (CE)')
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
            out_file = outfile + '_basin_{}'.format(basin)  + '_fluxes.' + out_format
            print "  - writing image %s ..." % out_file
            fig.savefig(out_file, bbox_inches='tight', dpi=out_res)


def plot_cumulative_fluxes_by_basin(plot_vars=['ice_mass', 'discharge_cumulative', 'surface_mass_flux_cumulative']):
    '''
    Make a plot per basin with all flux_plot_vars
    '''
    
    for k, ifile in enumerate(ifiles):

        fig = plt.figure()
        offset = transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)
        ax = fig.add_subplot(111)

        basin = basin_list[k]
        print('reading {}'.format(ifile))
        nc = NC(ifile, 'r')
        t = nc.variables["time"][:]

        date = np.arange(start_year + step,
                         start_year + (len(t[:]) + 1) * step,
                         step) 
        idx = np.where(np.array(date) == time_bounds[-1])[0][0]

        for mvar in plot_vars:
            if mvar=='discharge_cumulative':
                d_var_vals_sum = 0
                for d_var in ('discharge_cumulative', 'basal_mass_flux_cumulative'):
                    d_var_vals = -np.squeeze(nc.variables[d_var][:]) * gt2mSLE
                    iunits = nc.variables[d_var].units
                    d_var_vals_sum += unit_converter(d_var_vals, iunits, mass_ounits)
                var_vals = d_var_vals_sum
            else:
                var_vals = np.squeeze(nc.variables[mvar][:])
                iunits = nc.variables[mvar].units
                var_vals = unit_converter(var_vals, iunits, mass_ounits)
            if runmean is not None:
                runmean_var_vals = smooth(var_vals, window_len=runmean)
                plt.plot(date[:], var_vals[:],
                         color=basin_col_dict[basin],
                         lw=0.25,
                         ls=mass_style_dict[mvar])
                plt.plot(date[:], runmean_var_vals[:],
                         color=basin_col_dict[basin],
                         lw=0.5,
                         ls=mass_style_dict[mvar],
                         label=mass_abbr_dict[mvar])
            else:
                plt.plot(date[:], var_vals[:],
                         color=basin_col_dict[basin],
                         lw=0.5,
                         ls=mass_style_dict[mvar],
                         label=mass_abbr_dict[mvar])
        nc.close()

        if do_legend:
            legend = ax.legend(loc="upper right",
                               edgecolor='0',
                               bbox_to_anchor=(0, 0, 1.15, 1),
                               bbox_transform=plt.gcf().transFigure)
            legend.get_frame().set_linewidth(0.2)
    
        ax.set_xlabel('Year (CE)')
        ax.set_ylabel('cumulative mass change (Gt)')
        axSLE = ax.twinx()
        ax.set_autoscalex_on(False)
        axSLE.set_autoscalex_on(False)
            

        ymin, ymax = ax.get_ylim()
        # Plot twin axis on the right, in mmSLE
        yminSLE = ymin * gt2mmSLE
        ymaxSLE = ymax * gt2mmSLE
        axSLE.set_ylim(yminSLE, ymaxSLE)
        axSLE.set_ylabel('mm SLE')

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
            out_file = outfile + '_basin_{}'.format(basin)  + '_cumulative.' + out_format
            print "  - writing image %s ..." % out_file
            fig.savefig(out_file, bbox_inches='tight', dpi=out_res)


def plot_mass(plot_vars=mass_plot_vars):
    '''
    Plot mass time series for all basins in one plot
    '''
    
    fig = plt.figure()
    offset = transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)
    ax = fig.add_subplot(111)

    for ifile in ifiles:
        print('reading {}'.format(ifile))
        nc = NC(ifile, 'r')
        t = nc.variables["time"][:]

        date = np.arange(start_year + step,
                         start_year + (len(t[:]) + 1) * step,
                         step) 

        for mvar in plot_vars:
            var_vals = np.squeeze(nc.variables[mvar][:])
            iunits = nc.variables[mvar].units
            var_vals = unit_converter(var_vals, iunits, mass_ounits)
            # plot anomalies
            plt.plot(date[:], (var_vals[:] - var_vals[0]),
                     color=basin_col_dict[basin],
                     lw=0.5,
                     label=basin)
        nc.close()

    if do_legend:
        legend = ax.legend(loc="upper right",
                           edgecolor='0',
                           bbox_to_anchor=(0, 0, 1.15, 1),
                           bbox_transform=plt.gcf().transFigure)
        legend.get_frame().set_linewidth(0.2)

    axSLE = ax.twinx()
    ax.set_autoscalex_on(False)
    axSLE.set_autoscalex_on(False)
    
    ax.set_xlabel('Year (CE)')
    ax.set_ylabel('cumulative mass change (Gt)')
        
    if time_bounds:
        ax.set_xlim(time_bounds[0], time_bounds[1])

    if bounds:
        ax.set_ylim(bounds[0], bounds[1])

    ymin, ymax = ax.get_ylim()
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


def plot_flux_all_basins(mvar='tendency_of_ice_mass_due_to_discharge'):
    '''
    Plot discharge flux for all basins in one plot
    '''
    
    fig = plt.figure()
    ax = fig.add_subplot(111)

    for k, ifile in enumerate(ifiles):
        basin = basin_list[k]
        print('reading {}'.format(ifile))
        nc = NC(ifile, 'r')
        t = nc.variables["time"][:]

        date = np.arange(start_year + step,
                         start_year + (len(t[:]) + 1) * step,
                         step) 

        var_vals = np.squeeze(nc.variables[mvar][:])
        iunits = nc.variables[mvar].units
        var_vals = unit_converter(var_vals, iunits, flux_ounits)
        if runmean is not None:
            runmean_var_vals = smooth(var_vals, window_len=runmean)
            plt.plot(date[:], var_vals[:],
                     alpha=0.5,
                     color=basin_col_dict[basin],
                     lw=0.25,
                     ls='-')
            plt.plot(date[:], runmean_var_vals[:],
                     color=basin_col_dict[basin],
                     lw=0.5,
                     ls='-',
                     label=basin)
        else:
            plt.plot(date[:], var_vals[:],
                     color=basin_col_dict[basin],
                     lw=0.5,
                     ls='-',
                     label=basin)
        nc.close()

    if do_legend:
        legend = ax.legend(loc="upper right",
                           edgecolor='0',
                           bbox_to_anchor=(0, 0, 1.07, 0.9),
                           bbox_transform=plt.gcf().transFigure)
        legend.get_frame().set_linewidth(0.2)
        
    ax.set_xlabel('Year (CE)')
    ax.set_ylabel('mass flux (Gt yr$^{\mathregular{-1}}$)')
        
    if time_bounds:
        ax.set_xlim(time_bounds[0], time_bounds[1])

    if bounds:
        ax.set_ylim(bounds[0], bounds[1])

    xmin, xmax = ax.get_xlim()
    ax.hlines(0, xmin, xmax, lw=0.25)

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
        out_file = outfile + '_{}'.format(flux_short_dict[mvar])  + '.' + out_format
        print "  - writing image %s ..." % out_file
        fig.savefig(out_file, bbox_inches='tight', dpi=out_res)

def plot_basin_rel_discharge():

    fig = plt.figure()
    ax = fig.add_subplot(111)

    for k, ifile in enumerate(ifiles):
        basin = basin_list[k]
        print('reading {}'.format(ifile))
        nc = NC(ifile, 'r')
        t = nc.variables["time"][:]

        date = np.arange(start_year + step,
                         start_year + (len(t[:]) + 1) * step,
                         step) 

        idx = np.where(np.array(date) == time_bounds[-1])[0][0]

        mvar = 'tendency_of_ice_mass'
        mass_var_vals = np.squeeze(nc.variables[mvar][:])
        iunits = nc.variables[mvar].units
        mass_var_vals = unit_converter(mass_var_vals, iunits, flux_ounits)
        d_var_vals_sum = 0
        for d_var in ['tendency_of_ice_mass_due_to_discharge', 'tendency_of_ice_mass_due_to_basal_mass_flux']:
            d_var_vals = np.squeeze(nc.variables[d_var][:])
            iunits = nc.variables[d_var].units
            d_var_vals_sum += unit_converter(d_var_vals, iunits, flux_ounits)

        runmean = 10
        runmean_m = smooth(mass_var_vals, window_len=runmean)
        runmean_d = smooth(d_var_vals_sum, window_len=runmean)
            
        ax.plot(date[:],   runmean_d / runmean_m,
                color=basin_col_dict[basin],
                linewidth=0.35,
                label=basin)
        # ax.plot(date[:],  d_var_vals_sum[:],
        #         ':',
        #         color=basin_col_dict[basin],
        #         linewidth=0.35)
        nc.close()

    ax.hlines(0, time_bounds[0], time_bounds[-1], lw=0.25)

    if do_legend:
        legend = ax.legend(loc="upper right",
                           edgecolor='0',
                           bbox_to_anchor=(0, 0, 1.15, 1),
                           bbox_transform=plt.gcf().transFigure)
        legend.get_frame().set_linewidth(0.2)
    
    ax.set_xlabel('Year (CE)')
    ax.set_ylabel('rel.')
        
    if time_bounds:
        ax.set_xlim(time_bounds[0], time_bounds[1])

    if bounds:
        ax.set_ylim(bounds[0], bounds[-1])

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
        print "  - writing image %s ..." % out_file
        fig.savefig(out_file, bbox_inches='tight', dpi=out_res)

        
def plot_rel_discharge_flux_all_basins(mvar='tendency_of_ice_mass_due_to_discharge'):
    
    fig = plt.figure()
    offset = transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)
    ax = fig.add_subplot(111)

    for k, ifile in enumerate(ifiles):
        basin = basin_list[k]
        print('reading {}'.format(ifile))
        nc = NC(ifile, 'r')
        t = nc.variables["time"][:]

        date = np.arange(start_year + step,
                         start_year + (len(t[:]) + 1) * step,
                         step) 

        var_vals = np.squeeze(nc.variables[mvar][:]) * 100 - 100
        ax.plot(date[:], (var_vals[:]),
                 color=basin_col_dict[basin],
                 lw=0.75,
                 label=basin)
        nc.close()

    if do_legend:
        legend = ax.legend(loc="upper right",
                           edgecolor='0',
                           bbox_to_anchor=(0, 0, 1.15, 1),
                           bbox_transform=plt.gcf().transFigure)
        legend.get_frame().set_linewidth(0.2)
    
    ax.set_xlabel('Year (CE)')
    ax.set_ylabel('mass flux anomaly (%)')
        
    if time_bounds:
        ax.set_xlim(time_bounds[0], time_bounds[1])

    if bounds:
        ax.set_ylim(bounds[0], bounds[1])

    xmin, xmax = ax.get_xlim()
    ax.hlines(0, xmin, xmax, lw=0.25)

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
        out_file = outfile + '_discharge_flux_anomaly'  + '.' + out_format
        print "  - writing image %s ..." % out_file
        fig.savefig(out_file, bbox_inches='tight', dpi=out_res)

        
def plot_basin_mass(plot_var='ice_mass'):
    
    fig = plt.figure()
    offset = transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)
    ax = fig.add_subplot(111)

    my_var_vals = 0
    print('cumulative SLE {}m'.format(my_var_vals))
    for k, ifile in enumerate(ifiles):
        basin = basin_list[k]
        print('reading {}'.format(ifile))
        nc = NC(ifile, 'r')
        t = nc.variables["time"][:]

        date = np.arange(start_year + step,
                         start_year + (len(t[:]) + 1) * step,
                         step) 

        mvar = plot_var
        if plot_var == 'ice_mass':
            var_vals = -np.squeeze(nc.variables[mvar][:]) * gt2mSLE
            iunits = nc.variables[mvar].units
            var_vals = unit_converter(var_vals, iunits, mass_ounits)
        elif plot_var == 'discharge_cumulative':
            d_var_vals = 0
            for d_var in ('discharge_cumulative', 'basal_mass_flux_cumulative'):
                tmp_d_var_vals = -np.squeeze(nc.variables[d_var][:]) * gt2mSLE
                iunits = nc.variables[d_var].units
                d_var_vals += unit_converter(tmp_d_var_vals, iunits, mass_ounits)
            var_vals = d_var_vals
        else:
            pass    
        # plot anomalies
        ax.fill_between(date[:], my_var_vals, my_var_vals + var_vals[:],
                        color=basin_col_dict[basin],
                        linewidth=0,
                        label=basin)
        offset = 10
        try:
            x_sle, y_sle = time_bounds[-1] + offset, my_var_vals[-1]
        except:  # first iteration
            x_sle, y_sle = time_bounds[-1] + offset, my_var_vals
        print('basin {}: {}'.format(basin, var_vals[-1]))
        plt.text( x_sle, y_sle, '{: 1.2f}'.format(var_vals[-1]),
                          color=basin_col_dict[basin])
        nc.close()
        my_var_vals += var_vals
        print('cumulative SLE {}m'.format(my_var_vals[-1]))

    legend = ax.legend(loc="upper right",
                       edgecolor='0',
                       bbox_to_anchor=(0, 0, 1.15, 1),
                       bbox_transform=plt.gcf().transFigure)
    legend.get_frame().set_linewidth(0.2)
    
    ax.set_xlabel('Year (CE)')
    ax.set_ylabel('$\Delta$(GMSL) (m)')
        
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
        out_file = outfile + '_' + mvar  + '.' + out_format
        print "  - writing image %s ..." % out_file
        fig.savefig(out_file, bbox_inches='tight', dpi=out_res)

def plot_basin_mass_d():
    
    fig = plt.figure()
    offset = transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)
    ax = fig.add_subplot(111)

    mass_var_vals_positive_cum = 0
    mass_var_vals_negative_cum = 0
    for k, ifile in enumerate(ifiles):
        basin = basin_list[k]
        print('reading {}'.format(ifile))
        nc = NC(ifile, 'r')
        t = nc.variables["time"][:]

        date = np.arange(start_year + step,
                         start_year + (len(t[:]) + 1) * step,
                         step) 

        idx = np.where(np.array(date) == time_bounds[-1])[0][0]

        mvar = 'ice_mass'
        mass_var_vals = -np.squeeze(nc.variables[mvar][:]) * gt2mSLE
        iunits = nc.variables[mvar].units
        mass_var_vals = unit_converter(mass_var_vals, iunits, mass_ounits)
        if mass_var_vals[idx] > 0:
            ax.fill_between(date[:], mass_var_vals_positive_cum, mass_var_vals_positive_cum + mass_var_vals[:],
                            color=basin_col_dict[basin],
                            linewidth=0,
                            label=basin)
        else:
            print mass_var_vals[idx]
            ax.fill_between(date[:], mass_var_vals_negative_cum, mass_var_vals_negative_cum + mass_var_vals[:],
                            color=basin_col_dict[basin],
                            linewidth=0,
                            label=basin)
            plt.rcParams['hatch.color'] = basin_col_dict[basin]
            plt.rcParams['hatch.linewidth'] = 0.1
            ax.fill_between(date[:], mass_var_vals_negative_cum, mass_var_vals_negative_cum + mass_var_vals[:],
                            facecolor="none", hatch="XXXXX", edgecolor="k",
                            linewidth=0.0)
        d_var_vals_sum = 0
        for d_var in ['discharge_cumulative', 'basal_mass_flux_cumulative']:
            d_var_vals = -np.squeeze(nc.variables[d_var][:]) * gt2mSLE
            iunits = nc.variables[d_var].units
            d_var_vals_sum += unit_converter(d_var_vals, iunits, mass_ounits)
        # if mass_var_vals[idx] > 0:
        #     ax.fill_between(date[:], mass_var_vals_positive_cum, mass_var_vals_positive_cum + d_var_vals_sum[:],
        #                     color='0.8',
        #                     alpha=0.5,
        #                     linewidth=0)

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
            except:  # first iteratio
                x_sle, y_sle = date[idx] + offset, mass_var_vals_positive_cum
        else:
            try:
                x_sle, y_sle = date[idx] + offset, mass_var_vals_negative_cum[idx] + mass_var_vals[idx] 
            except:  # first iteration
                x_sle, y_sle = date[idx] + offset, mass_var_vals_negative_cum + mass_var_vals[idx] 
        # contribution of cumulative discharge to cumulative mass loss
        d_to_mass_percent = d_var_vals_sum[idx] / mass_var_vals[idx] * 100
        print basin, d_to_mass_percent ,d_var_vals_sum[idx], mass_var_vals[idx]
        if d_to_mass_percent > 0:
            plt.text( x_sle, y_sle, '{: 3.0f}%'.format(d_to_mass_percent),
                      color=basin_col_dict[basin])
        # plt.text( x_sle, y_sle, '{: 1.2f}m'.format(mass_var_vals[idx]),
        #           color=basin_col_dict[basin])
        nc.close()
        if mass_var_vals[idx] > 0:
            mass_var_vals_positive_cum += mass_var_vals
        else:
            mass_var_vals_negative_cum += mass_var_vals

    ax.hlines(0, time_bounds[0], time_bounds[-1], lw=0.25)

    legend = ax.legend(loc="upper right",
                       edgecolor='0',
                       bbox_to_anchor=(0, 0, 1.15, 1),
                       bbox_transform=plt.gcf().transFigure)
    legend.get_frame().set_linewidth(0.2)
    
    ax.set_xlabel('Year (CE)')
    ax.set_ylabel('$\Delta$(GMSL) (m)')
        
    if time_bounds:
        ax.set_xlim(time_bounds[0], time_bounds[1])

    if bounds:
        ax.set_ylim(bounds[0], bounds[-1])

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
        print "  - writing image %s ..." % out_file
        fig.savefig(out_file, bbox_inches='tight', dpi=out_res)

if plot == 'basin_discharge':
    plot_flux_all_basins()
elif plot == 'basin_smb':
    plot_flux_all_basins(plot_vars='tendency_of_ice_mass_due_to_surface_mass_flux')
elif plot == 'rel_basin_discharge':
    plot_rel_discharge_flux_all_basins()
elif plot == 'fluxes':
    plot_fluxes(plot_vars=['mass_rate_of_change_glacierized', 'discharge_flux','sub_shelf_ice_flux', 'surface_runoff_mass_flux', 'surface_accumulation_mass_flux'])
elif plot == 'basin_mass':
    plot_basin_mass()
elif plot == 'basin_mass_d':
    plot_basin_mass_d()
elif plot == 'basin_d_cumulative':
    plot_basin_mass(plot_var='discharge_cumulative')
elif plot == 'per_basin_fluxes':
    plot_fluxes_by_basin()
elif plot == 'per_basin_cumulative':
    plot_cumulative_fluxes_by_basin()
elif plot == 'rcp_ens_area':
    plot_rcp_cold()
elif plot == 'rcp_ens_volume':
    plot_rcp_cold(plot_var='rel_volume_cold')
elif plot == 'rcp_mass':
    plot_rcp_mass(plot_var='ice_mass')
elif plot == 'rcp_ens_d_flux':
    plot_rcp_ens_flux(plot_var='discharge')
elif plot == 'rcp_ens_smb_flux':
    plot_rcp_ens_flux(plot_var='smb')
elif plot == 'rcp_ens_mass_flux':
    plot_rcp_ens_flux(plot_var='mass')
elif plot == 'ens_mass':
    plot_ens_mass(plot_var='ice_mass')
elif plot == 'rcp_ens_mass':
    plot_rcp_ens_mass(plot_var='ice_mass')
elif plot == 'anim_rcp_mass':
    anim_rcp_mass(plot_var='ice_mass')
elif plot == 'rcp_flux':
    plot_rcp_flux()
elif plot == 'rcp_flux_rel':
    plot_rcp_flux_relative()
elif plot == 'rcp_lapse_mass':
    plot_rcp_lapse_mass(plot_var='ice_mass')
elif plot == 'rcp_d':
    plot_rcp_mass(plot_var='discharge_cumulative')
elif plot == 'flood_gate_length':
    plot_flood_gate_length_ts()
elif plot == 'flood_gate_area':
    plot_flood_gate_area_ts()
elif plot == 'basin_rel_discharge':
    plot_basin_rel_discharge()
