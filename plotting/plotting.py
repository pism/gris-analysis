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
    from pypismtools import unit_converter
except:
    from pypismtools.pypismtools import unit_converter

prefix = "les_gcm"


def input_filename(prefix, rcp, year):
    return "2018_09_les/sobol/{prefix}_rcp{rcp}_{year}_sobel.txt".format(prefix=prefix, rcp=rcp, year=year)


def read_sobel_file(filename):
    print(filename)
    data = np.loadtxt(filename, usecols=(1,))
    return data


def set_size(w, h, ax=None):
    """ w, h: width, height in inches """

    if not ax:
        ax = plt.gca()
    l = ax.figure.subplotpars.left
    r = ax.figure.subplotpars.right
    t = ax.figure.subplotpars.top
    b = ax.figure.subplotpars.bottom
    figw = float(w) / (r - l)
    figh = float(h) / (t - b)
    ax.figure.set_size_inches(figw, figh)


def gmtColormap(fileName, log_color=False, reverse=False):
    """
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
    """
    import colorsys
    import os

    try:
        try:
            f = open(fileName)
        except:
            # Check if it's a colormap provided in colormaps/
            basedir, fname = os.path.split(__file__)
            my_file = os.path.join(basedir, "colormaps", fileName)
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
            rr, gg, bb = colorsys.hsv_to_rgb(r[i] / 360.0, g[i], b[i])
            r[i] = rr
            g[i] = gg
            b[i] = bb
    if colorModel == "HSV":
        for i in range(r.shape[0]):
            rr, gg, bb = colorsys.hsv_to_rgb(r[i] / 360.0, g[i], b[i])
            r[i] = rr
            g[i] = gg
            b[i] = bb
    if colorModel == "RGB":
        r = r / 255.0
        g = g / 255.0
        b = b / 255.0

    if log_color:
        xNorm = np.zeros((len(x),))
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
    colorDict = {"red": red, "green": green, "blue": blue}
    return colorDict


basin_list = ["CW", "NE", "NO", "NW", "SE", "SW"]
rcp_list = ["26", "45", "85"]

# Set up the option parser
parser = ArgumentParser()
parser.description = "A script for PISM output files to time series plots using pylab/matplotlib."
parser.add_argument("FILE", nargs="*")
parser.add_argument(
    "--bounds", dest="bounds", nargs=2, type=float, help="lower and upper bound for ordinate, eg. -1 1", default=None
)
parser.add_argument(
    "--time_bounds",
    dest="time_bounds",
    nargs=2,
    type=float,
    help="lower and upper bound for abscissa, eg. 1990 2000",
    default=[2008, 3008],
)
parser.add_argument("-c", "--colormap", dest="my_colormap", help="Colormap for flowline plots", default="jet")
parser.add_argument(
    "-l",
    "--labels",
    dest="labels",
    help="comma-separated list with labels, put in quotes like 'label 1,label 2'",
    default=None,
)
parser.add_argument(
    "-f",
    "--output_format",
    dest="out_formats",
    help="Comma-separated list with output graphics suffix, default = pdf",
    default="pdf",
)
parser.add_argument(
    "-n",
    "--parallel_threads",
    dest="openmp_n",
    help="Number of OpenMP threads for operators such as enssstat, Default=1",
    default=1,
)
parser.add_argument("--no_legend", dest="do_legend", action="store_false", help="Do not plot legend", default=True)
parser.add_argument(
    "-o",
    "--output_file",
    dest="outfile",
    help="output file name without suffix, i.e. ts_control -> ts_control_variable",
    default="unnamed",
)
parser.add_argument(
    "--step", dest="step", type=int, help="step for plotting values, if time-series is very long", default=1
)
parser.add_argument("--start_year", dest="start_year", type=float, help="""Start year""", default=2008)
parser.add_argument(
    "--rotate_xticks",
    dest="rotate_xticks",
    action="store_true",
    help="rotate x-ticks by 30 degrees, Default=False",
    default=False,
)
parser.add_argument(
    "-r",
    "--output_resolution",
    dest="out_res",
    help="""Resolution ofoutput graphics in dots per
                  inch (DPI), default = 300""",
    type=int,
    default=300,
)
parser.add_argument(
    "--plot",
    dest="plot",
    help="""What to plot.""",
    choices=[
        "cmip5_rcp",
        "les",
        "ens_mass",
        "ctrl_mass",
        "forcing_mass",
        "mass_d",
        "profile",
        "profile_anim",
        "flux_partitioning",
        "basin_flux_partitioning",
        "ctrl_mass_anim",
        "d_contrib_anim",
        "grid_res",
        "pdfs",
        "random_flux",
        "sobel",
    ],
    default="les",
)

parser.add_argument("--title", dest="title", help="""Plot title.""", default=None)
parser.add_argument("--ctrl_file", dest="ctrl_file", nargs="*", help="""Filename of ctrl run""", default=None)

options = parser.parse_args()
my_colormap = options.my_colormap
ifiles = options.FILE
if options.labels != None:
    labels = options.labels.split(",")
else:
    labels = None
bounds = options.bounds
do_legend = options.do_legend
time_bounds = options.time_bounds
openmp_n = options.openmp_n
out_res = options.out_res
outfile = options.outfile
out_formats = options.out_formats.split(",")
plot = options.plot
rotate_xticks = options.rotate_xticks
step = options.step
title = options.title
ctrl_file = options.ctrl_file

if openmp_n > 1:
    pthreads = "-P {}".format(openmp_n)
else:
    pthreads = ""

dx, dy = 4.0 / out_res, -4.0 / out_res

# Conversion between giga tons (Gt) and millimeter sea-level equivalent (mmSLE)
gt2mmSLE = 1.0 / 365
gt2cmSLE = 1.0 / 365 / 10.0
gt2mSLE = 1.0 / 365 / 1000.0

start_year = options.start_year

# Plotting styles

fontsize = 6
lw = 0.65
aspect_ratio = 0.35
markersize = 2
fig_width = 3.1  # inch
fig_height = aspect_ratio * fig_width  # inch
fig_size = [fig_width, fig_height]

params = {
    "backend": "ps",
    "axes.linewidth": 0.25,
    "lines.linewidth": lw,
    "axes.labelsize": fontsize,
    "font.size": fontsize,
    "xtick.direction": "in",
    "xtick.labelsize": fontsize,
    "xtick.major.size": 2.5,
    "xtick.major.width": 0.25,
    "ytick.direction": "in",
    "ytick.labelsize": fontsize,
    "ytick.major.size": 2.5,
    "ytick.major.width": 0.25,
    "legend.fontsize": fontsize,
    "lines.markersize": markersize,
    "font.size": fontsize,
    "figure.figsize": fig_size,
}

plt.rcParams.update(params)


rcp_col_dict = {"CTRL": "k", "85": "#990002", "45": "#5492CD", "26": "#003466"}

rcp_shade_col_dict = {"CTRL": "k", "85": "#F4A582", "45": "#92C5DE", "26": "#4393C3"}

rcp_dict = {"26": "RCP 2.6", "45": "RCP 4.5", "85": "RCP 8.5", "CTRL": "CTRL"}

line_colors = ["#5492CD", "#C47900", "#004F00", "#003466", "#808080"]

res_col_dict = {
    "450": "#006d2c",
    "600": "#31a354",
    "900": "#74c476",
    "1800": "#bae4b3",
    "3600": "#fcae91",
    "4500": "#fb6a4a",
    "9000": "#de2d26",
    "18000": "#a50f15",
}


area_ounits = "m2"
flux_ounits = "Gt year-1"
specific_flux_ounits = "kg m-2 year-1"
mass_ounits = "Gt"
mass_plot_var = "limnsw"
flux_plot_var = "tendendy_of_ice_mass_due_to_discharge"

runmean_window = 11


def add_inner_title(ax, title, loc, size=None, **kwargs):
    """
    Adds an inner title to a given axis, with location loc.

    from http://matplotlib.sourceforge.net/examples/axes_grid/demo_axes_grid2.html
    """
    from matplotlib.offsetbox import AnchoredText
    from matplotlib.patheffects import withStroke

    prop = dict(size=plt.rcParams["legend.fontsize"], weight="bold")
    at = AnchoredText(title, loc=loc, prop=prop, pad=0.0, borderpad=0.5, frameon=False, **kwargs)
    ax.add_artist(at)
    return at


def plot_cmip5_rcp(plot_var="delta_T"):

    fig, ax = plt.subplots(3, 1, sharex="col", figsize=[6, 4])
    fig.subplots_adjust(hspace=0.25, wspace=0.05)

    for k, rcp in enumerate(rcp_list[:]):

        rcp_files = [f for f in ifiles if "rcp{}".format(rcp) in f]

        gcm_files = [f for f in rcp_files if ("r1i1p1" in f) and not ("ENS" in f)]
        ensstdm1_file = [f for f in rcp_files if "ensstdm1" in f][0]
        ensstdp1_file = [f for f in rcp_files if "ensstdp1" in f][0]
        ensstdm1_cdf = cdo.readCdf(ensstdm1_file)
        ensstdp1_cdf = cdo.readCdf(ensstdp1_file)
        t = ensstdp1_cdf.variables["time"][:]
        cmip5_date = np.arange(start_year + step, start_year + (len(t[:]) + 1) * step, step)

        ensstdm1_vals = np.squeeze(ensstdm1_cdf.variables[plot_var][:])
        ensstdp1_vals = np.squeeze(ensstdp1_cdf.variables[plot_var][:])
        ax[k].fill_between(cmip5_date, ensstdm1_vals, ensstdp1_vals, alpha=0.20, linewidth=0.50, color="k")

        for m, gcm_file in enumerate(gcm_files):
            gcm = gcm_file.split("tas_Amon_")[1].split("_rcp")[0]
            gcm_cdf = cdo.readCdf(gcm_file)
            t = gcm_cdf.variables["time"][:]
            gcm_date = np.arange(start_year + step, start_year + (len(t[:]) + 1) * step, step)
            gcm_vals = np.squeeze(gcm_cdf.variables[plot_var][:])
            ax[k].plot(gcm_date, gcm_vals, label=gcm, color=line_colors[m], linewidth=0.4)

        ensmean_file = [f for f in rcp_files if ("r1i1p1" in f) and ("ENSMEAN" in f)][0]
        ensmean_cdf = cdo.readCdf(ensmean_file)
        t = ensmean_cdf.variables["time"][:]
        ensmean_date = np.arange(start_year + step, start_year + (len(t[:]) + 1) * step, step)
        ensmean_vals = np.squeeze(ensmean_cdf.variables[plot_var][:])
        gcm = ensmean_file.split("tas_Amon_")[1].split("_rcp")[0]
        ax[k].plot(ensmean_date, ensmean_vals, label=gcm, color="k", linewidth=lw)

        if time_bounds:
            ax[k].set_xlim(time_bounds[0], time_bounds[1])

        if bounds:
            ax[k].set_ylim(bounds[0], bounds[1])

        ymin, ymax = ax[k].get_ylim()
        ax[k].yaxis.set_major_formatter(FormatStrFormatter("%1.0f"))

        ax[k].set_title("{}".format(rcp_dict[rcp]))

    ax[2].set_xlabel("Year")
    ax[1].set_ylabel("T-anomaly (K)")

    if do_legend:
        legend = ax[2].legend(
            loc="lower left", edgecolor="0", bbox_to_anchor=(0.13, 0.16, 0, 0), bbox_transform=plt.gcf().transFigure
        )
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
        out_file = outfile + "_cmip5_" + plot_var + "." + out_format
        print("  - writing image %s ..." % out_file)
        fig.savefig(out_file, bbox_inches="tight", dpi=out_res)


def plot_profile_ts_combined():

    try:
        cmap = getattr(plt.cm, my_colormap)
    except:
        # import and convert colormap
        cdict = gmtColormap(my_colormap)
        cmap = mpl.colors.LinearSegmentedColormap("my_colormap", cdict, 1024)

    mcm = cm = plt.get_cmap(cmap)

    nc = NC(ifiles[0], "r")
    profile_names = nc.variables["profile_name"][:]
    for k, profile in enumerate(profile_names):

        print((u"Processing {} profile".format(profile)))

        fig, ax = plt.subplots(3, 1, sharex="col", figsize=[3, 2])
        fig.subplots_adjust(hspace=0.15, wspace=0.05)

        profile_iunits = nc.variables["profile"].units
        profile_ounits = "km"
        profile_vals = nc.variables["profile"][k, :]
        profile_vals = unit_converter(profile_vals, profile_iunits, profile_ounits)

        t_var = nc.variables["time"][:]
        date = np.arange(start_year, start_year + (len(t_var[:]) + 1), step)
        ma = np.where(date == time_bounds[0])[0][0]
        me = np.where(date == time_bounds[1])[0][0]

        plot_times = np.arange(ma, me + 1, step)

        cNorm = colors.Normalize(vmin=time_bounds[0], vmax=time_bounds[1])
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=mcm)
        speed_vals_0 = np.nanmean(nc.variables["velsurf_mag"][k, 0:20, :], axis=0)

        for t in plot_times:
            colorVal = scalarMap.to_rgba(date[t])
            speed_vals = nc.variables["velsurf_mag"][k, t, :] - speed_vals_0
            speed_vals = nc.variables["velsurf_mag"][k, t, :]
            # speed_basal_vals = nc.variables['velbase_mag'][k, t, :]
            mask = speed_vals < 1
            speed_vals = np.ma.array(speed_vals, mask=mask)
            # speed_basal_vals = np.ma.array(speed_basal_vals, mask = mask)
            # slip_ratio = speed_vals / speed_basal_vals
            topg_vals = nc.variables["topg"][k, t, :]
            thk_vals = nc.variables["thk"][k, t, :]
            usurf_vals = nc.variables["usurf"][k, t, :]
            thk_mask = thk_vals <= 20
            thk_vals = np.ma.array(thk_vals, mask=thk_mask)
            usurf_mask = usurf_vals < 100
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
                ax[2].plot(profile_vals, topg_vals, color="k", linewidth=0.3)
                try:
                    ax[2].plot(profile_vals, usurf_vals, color="k", linewidth=0.1)
                    bottom_vals = np.maximum(usurf_vals - thk_vals, topg_vals)
                    ax[2].plot(profile_vals, np.ma.array(bottom_vals, mask=thk_mask), color="k", linewidth=0.1)
                except:
                    pass

        xmin, xmax = ax[1].get_xlim()
        ymin, ymax = ax[1].get_ylim()

        ymin = -1200

        ax[0].set_ylabel("Flux\n (km$^{\mathregular{2}}$ yr$^{\mathregular{-1}}$)", multialignment="center")
        ax[1].set_ylabel("Speed\n (m yr$^{\mathregular{-1}}$)", multialignment="center")
        ax[2].fill_between([xmin, xmax], [ymin, ymin], color="#c6dbef", linewidth=0)
        tz = ax[2].fill_between(profile_vals, topg_vals * 0 + ymin, topg_vals, color="#fdbe85", linewidth=0)
        ax[2].set_ylabel("Altitude\n (masl)", multialignment="center")
        tz = ax[2].axhline(profile_vals[0], linestyle="solid", color="k", linewidth=0.3)
        tz.set_zorder(-1)
        ax[2].set_xlabel("distance ({})".format(profile_ounits))

        ax[2].set_xlim(np.nanmin(profile_vals), np.nanmax(profile_vals))
        ax[2].set_ylim(np.nanmin(topg_vals))
        ax[2].yaxis.set_major_formatter(FormatStrFormatter("%1.0f"))

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
            out_file = outfile + "_{}".format(unidecode(profile).replace(" ", "_")) + "." + out_format
            print("  - writing image %s ..." % out_file)
            fig.savefig(out_file, bbox_inches="tight", dpi=out_res)

    nc.close()


def plot_profile_ts_animation():

    try:
        cmap = getattr(plt.cm, my_colormap)
    except:
        # import and convert colormap
        cdict = gmtColormap(my_colormap)
        cmap = mpl.colors.LinearSegmentedColormap("my_colormap", cdict, 1024)

    mcm = cm = plt.get_cmap(cmap)

    nc = NC(ifiles[0], "r")
    profile_names = nc.variables["profile_name"][:]
    profile = u"Upernavik Isstr\xf8m S"
    k = 1

    print((u"Processing {} profile".format(profile)))

    profile_iunits = nc.variables["profile"].units
    profile_ounits = "km"
    profile_vals = nc.variables["profile"][k, :]
    profile_vals = unit_converter(profile_vals, profile_iunits, profile_ounits)

    t_var = nc.variables["time"][:]
    date = np.arange(start_year, start_year + (len(t_var[:]) + 1), step)
    ma = np.where(date == time_bounds[0])[0][0]
    me = np.where(date == time_bounds[1])[0][0]

    plot_times = np.arange(ma, me + 1, step)

    for t in plot_times:

        fig, ax = plt.subplots(3, 1, sharex="col", figsize=[3, 2])
        fig.subplots_adjust(hspace=0.15, wspace=0.05)

        cNorm = colors.Normalize(vmin=time_bounds[0], vmax=time_bounds[1])
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=mcm)

        # plot previous fluxes
        for ts in range(plot_times[0], t):
            colorVal = scalarMap.to_rgba(date[t])
            speed_vals = nc.variables["velsurf_mag"][k, ts, :]
            mask = speed_vals < 1
            speed_vals = np.ma.array(speed_vals, mask=mask)
            topg_vals = nc.variables["topg"][k, ts, :]
            thk_vals = nc.variables["thk"][k, ts, :]
            usurf_vals = nc.variables["usurf"][k, ts, :]
            thk_mask = thk_vals <= 20
            thk_vals = np.ma.array(thk_vals, mask=thk_mask)
            usurf_mask = usurf_vals < 100
            usurf_mask = np.logical_or((usurf_vals < topg_vals), thk_mask)
            usurf_vals = np.ma.array(usurf_vals, mask=usurf_mask)
            ax[0].plot(profile_vals, speed_vals * thk_vals / 1e6, color="0.5", linewidth=0.1)
            ax[1].plot(profile_vals, speed_vals, color="0.5", linewidth=0.1)

        colorVal = scalarMap.to_rgba(date[t])
        speed_vals = nc.variables["velsurf_mag"][k, t, :]
        mask = speed_vals < 1
        speed_vals = np.ma.array(speed_vals, mask=mask)
        topg_vals = nc.variables["topg"][k, t, :]
        thk_vals = nc.variables["thk"][k, t, :]
        usurf_vals = nc.variables["usurf"][k, t, :]
        thk_mask = thk_vals <= 20
        thk_vals = np.ma.array(thk_vals, mask=thk_mask)
        usurf_mask = usurf_vals < 100
        usurf_mask = np.logical_or((usurf_vals < topg_vals), thk_mask)
        usurf_vals = np.ma.array(usurf_vals, mask=usurf_mask)
        ax[0].plot(profile_vals, speed_vals * thk_vals / 1e6, color="k", linewidth=0.6)
        ax[1].plot(profile_vals, speed_vals, color="k", linewidth=0.6)
        try:
            ax[2].plot(profile_vals, usurf_vals, color="k", linewidth=0.3)
            bottom_vals = np.maximum(usurf_vals - thk_vals, topg_vals)
            ax[2].plot(profile_vals, np.ma.array(bottom_vals, mask=thk_mask), color="k", linewidth=0.3)
            ax[2].fill_between(
                profile_vals, np.ma.array(bottom_vals, mask=thk_mask), usurf_vals, color="#d9d9d9", linewidth=0.3
            )
            ax[2].fill_between(profile_vals, bottom_vals, usurf_vals, color="#bdbdbd", linewidth=0.3)
        except:
            pass
        if t == plot_times[-1]:
            ax[2].plot(profile_vals, topg_vals, color="k", linewidth=0.3)
            try:
                ax[2].plot(profile_vals, usurf_vals, color="k", linewidth=0.1)
                bottom_vals = np.maximum(usurf_vals - thk_vals, topg_vals)
                ax[2].plot(profile_vals, np.ma.array(bottom_vals, mask=thk_mask), color="k", linewidth=0.15)
            except:
                pass

        xmin, xmax = ax[1].get_xlim()
        ymin, ymax = ax[1].get_ylim()

        ymin = -1200
        ax[0].set_ylabel("Flux\n (km$^{\mathregular{2}}$ yr$^{\mathregular{-1}}$)", multialignment="center")
        ax[1].set_ylabel("Speed\n (m yr$^{\mathregular{-1}}$)", multialignment="center")
        tz = ax[2].fill_between([0, xmax], [ymin, ymin], color="#c6dbef", linewidth=0)
        tz.set_zorder(-1)
        tz = ax[2].fill_between(profile_vals, topg_vals * 0 + ymin, topg_vals, color="#fdbe85", linewidth=0)
        ax[2].set_ylabel("Altitude\n (masl)", multialignment="center")
        tz = ax[2].axhline(profile_vals[0], linestyle="solid", color="k", linewidth=0.3)
        tz.set_zorder(-1)
        ax[2].set_xlabel("distance ({})".format(profile_ounits))

        ax[2].set_xlim(np.nanmin(profile_vals), np.nanmax(profile_vals))
        ax[2].set_ylim(np.nanmin(topg_vals))
        ax[2].yaxis.set_major_formatter(FormatStrFormatter("%1.0f"))

        if bounds:
            ax[1].set_ylim(bounds[0], bounds[1])

        ax[0].set_ylim(0, 2.75)
        ax[2].set_ylim(-600, 2000)
        if time_bounds:
            ax[0].set_xlim(time_bounds[0], time_bounds[-1])
            ax[1].set_xlim(time_bounds[0], time_bounds[-1])
            ax[2].set_xlim(time_bounds[0], time_bounds[-1])
        else:
            ax[0].set_xlim(0, 100)
            ax[1].set_xlim(0, 100)
            ax[2].set_xlim(0, 100)

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
            out_file = outfile + "_{}_{:04d}".format(unidecode(profile).replace(" ", "_"), t) + "." + out_format
            print("  - writing image %s ..." % out_file)
            fig.savefig(out_file, bbox_inches="tight", dpi=out_res)
            plt.close()
            del fig

    nc.close()


def plot_ctrl_mass_anim(plot_var=mass_plot_var):

    rcp_26_file = [f for f in ifiles if "rcp_{}".format(26) in f][0]
    cdf_26 = cdo.readCdf(rcp_26_file)
    t_26 = cdf_26.variables["time"][:]
    date_26 = np.arange(start_year + step, start_year + (len(t_26[:]) + 1), step)
    var_vals_26 = cdf_26.variables[plot_var][:] - cdf_26.variables[plot_var][0]
    iunits = cdf_26.variables[plot_var].units
    var_vals_26 = -unit_converter(var_vals_26, iunits, mass_ounits) * gt2mSLE

    rcp_45_file = [f for f in ifiles if "rcp_{}".format(45) in f][0]
    cdf_45 = cdo.readCdf(rcp_45_file)
    t_45 = cdf_45.variables["time"][:]
    date_45 = np.arange(start_year + step, start_year + (len(t_45[:]) + 1), step)
    var_vals_45 = cdf_45.variables[plot_var][:] - cdf_45.variables[plot_var][0]
    iunits = cdf_45.variables[plot_var].units
    var_vals_45 = -unit_converter(var_vals_45, iunits, mass_ounits) * gt2mSLE

    rcp_85_file = [f for f in ifiles if "rcp_{}".format(85) in f][0]
    cdf_85 = cdo.readCdf(rcp_85_file)
    t_85 = cdf_85.variables["time"][:]
    date_85 = np.arange(start_year + step, start_year + (len(t_85[:]) + 1), step)
    var_vals_85 = cdf_85.variables[plot_var][:] - cdf_85.variables[plot_var][0]
    iunits = cdf_85.variables[plot_var].units
    var_vals_85 = -unit_converter(var_vals_85, iunits, mass_ounits) * gt2mSLE

    for frame in range(1000):

        fig = plt.figure()
        offset = transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)
        ax = fig.add_subplot(111)

        plt.plot(date_26, var_vals_26, color=rcp_col_dict["26"], linewidth=lw, alpha=0.5)

        plt.plot(date_26[:frame], var_vals_26[:frame], color=rcp_col_dict["26"], linewidth=lw, label=rcp_dict["26"])

        plt.plot(date_26[frame], var_vals_26[frame], color=rcp_col_dict["26"], marker="o", markersize=3, linewidth=0)

        plt.plot(date_45, var_vals_45, color=rcp_col_dict["45"], linewidth=lw, alpha=0.5)

        plt.plot(date_45[:frame], var_vals_45[:frame], color=rcp_col_dict["45"], linewidth=lw, label=rcp_dict["45"])

        plt.plot(date_45[frame], var_vals_45[frame], color=rcp_col_dict["45"], marker="o", markersize=3, linewidth=0)

        plt.plot(date_85, var_vals_85, color=rcp_col_dict["85"], linewidth=lw, alpha=0.5)

        plt.plot(date_85[:frame], var_vals_85[:frame], color=rcp_col_dict["85"], linewidth=lw, label=rcp_dict["85"])

        plt.plot(date_85[frame], var_vals_85[frame], color=rcp_col_dict["85"], marker="o", markersize=3, linewidth=0)

        legend = ax.legend(
            loc="center left", edgecolor="0", bbox_to_anchor=(0.15, 0.65), bbox_transform=plt.gcf().transFigure
        )
        legend.get_frame().set_linewidth(0.0)
        legend.get_frame().set_alpha(0.0)

        ax.set_xlabel("Year")
        ax.set_ylabel("sea level contribution (m)")

        if time_bounds:
            ax.set_xlim(time_bounds[0], time_bounds[1])

        if bounds:
            ax.set_ylim(bounds[0], bounds[1])

        ymin, ymax = ax.get_ylim()

        ax.yaxis.set_major_formatter(FormatStrFormatter("%1.2f"))

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
            out_file = "dgmsl_" + plot_var + "_{:04d}.{}".format(frame, "png")
            print("  - writing image %s ..." % out_file)
            fig.savefig(out_file, bbox_inches="tight", dpi=300)

        plt.close(fig)


def plot_d_contrib_anim(plot_var=mass_plot_var):

    for k, rcp in reversed(list(enumerate(rcp_list[::-1]))):

        rcp_files = [f for f in ifiles if "rcp_{}".format(rcp) in f]
        pctl16_file = [f for f in rcp_files if "enspctl16" in f]
        pctl84_file = [f for f in rcp_files if "enspctl84" in f]

        cdf_enspctl16 = cdo.runmean(runmean_window, input=pctl16_file, returnCdf=True, options=pthreads)
        cdf_enspctl84 = cdo.runmean(runmean_window, input=pctl84_file, returnCdf=True, options=pthreads)
        t = cdf_enspctl16.variables["time"][:]
        date = np.arange(start_year + step, start_year + (len(t[:]) + 1), step)

        enspctl16 = cdf_enspctl16.variables[plot_var][:]
        enspctl16_vals = cdf_enspctl16.variables[plot_var][:]

        enspctl84 = cdf_enspctl84.variables[plot_var][:]
        enspctl84_vals = cdf_enspctl84.variables[plot_var][:]

        for frame in range(1000):

            fig = plt.figure()
            offset = transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)
            ax = fig.add_subplot(111)

            ax.fill_between(date, enspctl84_vals, enspctl16_vals, color=rcp_col_dict[rcp], linewidth=lw, alpha=0.1)
            ax.plot(date[:frame], enspctl16_vals[:frame], color=rcp_col_dict[rcp], linewidth=lw)
            ax.plot(date[:frame], enspctl84_vals[:frame], color=rcp_col_dict[rcp], linewidth=lw)
            ax.fill_between(
                date[:frame], enspctl84_vals[:frame], enspctl16_vals[:frame], color=rcp_col_dict[rcp], linewidth=lw
            )

            ax.set_xlabel("Year")
            ax.set_ylabel("$\dot D_{\%}$ (%)")

            if time_bounds:
                ax.set_xlim(time_bounds[0], time_bounds[1])

            if bounds:
                ax.set_ylim(bounds[0], bounds[1])

            ymin, ymax = ax.get_ylim()

            ax.yaxis.set_major_formatter(FormatStrFormatter("%1.0f"))

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
                out_file = "d_contrib_" + plot_var + "_rcp{}_{:04d}.{}".format(rcp, frame, out_format)
                print("  - writing image %s ..." % out_file)
                fig.savefig(out_file, bbox_inches="tight", dpi=300)

            plt.close(fig)


def plot_grid_res(plot_var="tendency_of_ice_mass_due_to_discharge"):

    for k, rcp in enumerate(["45"]):

        fig = plt.figure()
        offset = transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)
        ax = fig.add_subplot(111)

        print(("Reading RCP {} files".format(rcp)))
        rcp_files = [f for f in ifiles if "rcp_{}".format(rcp) in f]

        for m_file in rcp_files:
            dr = re.search("gris_g(.+?)m", m_file).group(1)
            cdf = cdo.readCdf(m_file)

            t = cdf.variables["time"][:]

            vals = cdf.variables[plot_var][:]
            iunits = cdf[plot_var].units
            vals = unit_converter(vals, iunits, flux_ounits)

            date = np.arange(start_year + step, start_year + (len(t[:]) + 1), step)

            ax.plot(date[:], vals, color=res_col_dict[dr], alpha=0.5, linewidth=0.3)

        for m_file in rcp_files:
            dr = re.search("gris_g(.+?)m", m_file).group(1)
            cdf = cdo.runmean(runmean_window, input=m_file, returnCdf=True, options=pthreads)

            t = cdf.variables["time"][:]

            vals = cdf.variables[plot_var][:]
            iunits = cdf[plot_var].units
            vals = unit_converter(vals, iunits, flux_ounits)

            date = np.arange(start_year + step, start_year + (len(t[:]) + 1), step)

            ax.plot(date[:], vals, color=res_col_dict[dr], linewidth=lw, label=dr)

        ax.set_xlabel("Year")
        ax.set_ylabel("Rate (Gt yr$^{\mathregular{-1}}$)")

        if time_bounds:
            ax.set_xlim(time_bounds[0], time_bounds[1])

        if bounds:
            ax.set_ylim(bounds[0], bounds[1])

        ymin, ymax = ax.get_ylim()

        ax.yaxis.set_major_formatter(FormatStrFormatter("%1.0f"))

        if rotate_xticks:
            ticklabels = ax.get_xticklabels()
            for tick in ticklabels:
                tick.set_rotation(30)
        else:
            ticklabels = ax.get_xticklabels()
            for tick in ticklabels:
                tick.set_rotation(0)

        if do_legend:
            legend = ax.legend(
                loc="center right", edgecolor="0", bbox_to_anchor=(1.1, 0.5), bbox_transform=plt.gcf().transFigure
            )
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
            out_file = outfile + "_rcp_{}_grid".format(rcp) + "_" + plot_var + "." + out_format
            print("  - writing image %s ..." % out_file)
            fig.savefig(out_file, bbox_inches="tight", dpi=out_res)


def plot_les(plot_var=mass_plot_var):

    fig, ax = plt.subplots(5, 1, sharex="col", sharey="row", figsize=[3, 4])
    fig.subplots_adjust(hspace=0.05, wspace=0.30)

    print("Forcing")
    plot_var = "delta_T"
    for k, rcp in reversed(list(enumerate(rcp_list[::-1]))):

        rcp_files = [f for f in ifiles if "rcp{}".format(rcp) in f]

        ensmin_file = [f for f in rcp_files if ("ENSMIN" in f) and ("tas" in f)][0]
        ensmax_file = [f for f in rcp_files if ("ENSMAX" in f) and ("tas" in f)][0]
        ensmean_file = [f for f in rcp_files if ("ENSMEAN" in f) and ("tas" in f)][0]
        ensmin_cdf = cdo.readCdf(ensmin_file)
        ensmax_cdf = cdo.readCdf(ensmax_file)
        ensmean_cdf = cdo.readCdf(ensmean_file)

        t = ensmin_cdf.variables["time"][:]
        ensmin_date = np.arange(start_year + step, start_year + (len(t[:]) + 1) * step, step)

        t = ensmax_cdf.variables["time"][:]
        ensmax_date = np.arange(start_year + step, start_year + (len(t[:]) + 1) * step, step)

        t = ensmean_cdf.variables["time"][:]
        ensmean_date = np.arange(start_year + step, start_year + (len(t[:]) + 1) * step, step)

        ensmin_vals = np.squeeze(ensmin_cdf.variables[plot_var][:])
        ensmax_vals = np.squeeze(ensmax_cdf.variables[plot_var][:])
        ensmean_vals = np.squeeze(ensmean_cdf.variables[plot_var][:])

        ax[0].fill_between(ensmean_date, ensmin_vals, ensmax_vals, linewidth=0.0, color=rcp_shade_col_dict[rcp])

        ax[0].plot(ensmin_date, ensmin_vals, color=rcp_col_dict[rcp], linewidth=0.20)
        ax[0].plot(ensmax_date, ensmax_vals, color=rcp_col_dict[rcp], linewidth=0.20)
        ax[0].plot(ensmean_date, ensmean_vals, color=rcp_col_dict[rcp], label=rcp_dict[rcp], linewidth=lw)

    print("Cumulative Mass")
    plot_var = "limnsw"
    for k, rcp in enumerate(rcp_list[::-1]):

        print(("Reading RCP {} files".format(rcp)))
        rcp_files = [f for f in ifiles if ("rcp_{}".format(rcp) in f) and not ("tas" in f)]

        pctl16_file = [f for f in rcp_files if "enspctl16" in f][0]
        pctl84_file = [f for f in rcp_files if "enspctl84" in f][0]

        cdf_enspctl16 = cdo.readCdf(pctl16_file)
        cdf_enspctl84 = cdo.readCdf(pctl84_file)
        t = cdf_enspctl16.variables["time"][:]

        enspctl16 = cdf_enspctl16.variables[plot_var][:]
        enspctl16_vals = cdf_enspctl16.variables[plot_var][:] - cdf_enspctl16.variables[plot_var][0]
        iunits = cdf_enspctl16[plot_var].units
        enspctl16_vals = -unit_converter(enspctl16_vals, iunits, mass_ounits) * gt2mSLE

        enspctl84 = cdf_enspctl84.variables[plot_var][:]
        enspctl84_vals = cdf_enspctl84.variables[plot_var][:] - cdf_enspctl84.variables[plot_var][0]
        iunits = cdf_enspctl84[plot_var].units
        enspctl84_vals = -unit_converter(enspctl84_vals, iunits, mass_ounits) * gt2mSLE

        date = np.arange(start_year + step, start_year + (len(t[:]) + 1), step)

        # ensemble between 16th and 84th quantile
        ax[1].fill_between(date[:], enspctl16_vals, enspctl84_vals, color=rcp_shade_col_dict[rcp], linewidth=0)

        ax[1].plot(date[:], enspctl16_vals, color=rcp_col_dict[rcp], linestyle="solid", linewidth=0.20)

        ax[1].plot(date[:], enspctl84_vals, color=rcp_col_dict[rcp], linestyle="solid", linewidth=0.20)

        if ctrl_file is not None:
            rcp_ctrl_file = [f for f in ctrl_file if "rcp_{}".format(rcp) in f][0]

            cdf_ctrl = cdo.readCdf(rcp_ctrl_file)
            ctrl_t = cdf_ctrl.variables["time"][:]
            cdf_date = np.arange(start_year + step, start_year + (len(ctrl_t[:]) + 1), step)

            ctrl_vals = cdf_ctrl.variables[plot_var][:] - cdf_ctrl.variables[plot_var][0]
            iunits = cdf_ctrl[plot_var].units
            ctrl_vals = -unit_converter(ctrl_vals, iunits, mass_ounits) * gt2mSLE
            ax[1].plot(cdf_date[:], ctrl_vals, color=rcp_col_dict[rcp], linestyle="solid", linewidth=lw)

    print("Mass Loss Rate")
    plot_var = "tendency_of_ice_mass_glacierized"
    for k, rcp in enumerate(rcp_list[::-1]):

        print(("Reading RCP {} files".format(rcp)))
        rcp_files = [f for f in ifiles if ("rcp_{}".format(rcp) in f) and not ("tas" in f)]

        pctl16_file = [f for f in rcp_files if "enspctl16" in f][0]
        pctl84_file = [f for f in rcp_files if "enspctl84" in f][0]

        cdf_enspctl16 = cdo.runmean(runmean_window, input=pctl16_file, returnCdf=True, options=pthreads)
        cdf_enspctl84 = cdo.runmean(runmean_window, input=pctl84_file, returnCdf=True, options=pthreads)
        t = cdf_enspctl16.variables["time"][:]

        enspctl16 = cdf_enspctl16.variables[plot_var][:]
        enspctl16_vals = cdf_enspctl16.variables[plot_var][:] - cdf_enspctl16.variables[plot_var][0]
        iunits = cdf_enspctl16[plot_var].units
        enspctl16_vals = -unit_converter(enspctl16_vals, iunits, flux_ounits) * gt2mmSLE

        enspctl84 = cdf_enspctl84.variables[plot_var][:]
        enspctl84_vals = cdf_enspctl84.variables[plot_var][:] - cdf_enspctl84.variables[plot_var][0]
        iunits = cdf_enspctl84[plot_var].units
        enspctl84_vals = -unit_converter(enspctl84_vals, iunits, flux_ounits) * gt2mmSLE

        date = np.arange(start_year + step, start_year + (len(t[:]) + 1), step)

        # ensemble between 16th and 84th quantile
        ax[2].fill_between(date[:], enspctl16_vals, enspctl84_vals, color=rcp_shade_col_dict[rcp], linewidth=0)

        ax[2].plot(date[:], enspctl16_vals, color=rcp_col_dict[rcp], linestyle="solid", linewidth=0.20)

        ax[2].plot(date[:], enspctl84_vals, color=rcp_col_dict[rcp], linestyle="solid", linewidth=0.20)

        if ctrl_file is not None:
            rcp_ctrl_file = [f for f in ctrl_file if "rcp_{}".format(rcp) in f][0]

            cdf_ctrl = cdo.runmean(runmean_window, input=rcp_ctrl_file, returnCdf=True, options=pthreads)
            ctrl_t = cdf_ctrl.variables["time"][:]
            cdf_date = np.arange(start_year + step, start_year + (len(ctrl_t[:]) + 1), step)

            ctrl_vals = cdf_ctrl.variables[plot_var][:] - cdf_ctrl.variables[plot_var][0]
            iunits = cdf_ctrl[plot_var].units
            ctrl_vals = -unit_converter(ctrl_vals, iunits, flux_ounits) * gt2mmSLE
            ax[2].plot(cdf_date[:], ctrl_vals, color=rcp_col_dict[rcp], linestyle="solid", linewidth=lw)

    print("Discharge Contribution Absolute")
    plot_var = "discharge_contrib"
    end_years = [992, 992, 992]
    for k, rcp in enumerate(rcp_list[::-1]):

        rcp_files = [f for f in ifiles if ("rcp_{}".format(rcp) in f) and ("flux_absolute" in f)]

        pctl16_file = [f for f in rcp_files if ("enspctl16" in f)][0]
        pctl84_file = [f for f in rcp_files if ("enspctl84" in f)][0]

        cdf_enspctl16 = cdo.runmean(runmean_window, input=pctl16_file, returnCdf=True, options=pthreads)
        cdf_enspctl84 = cdo.runmean(runmean_window, input=pctl84_file, returnCdf=True, options=pthreads)
        t = cdf_enspctl16.variables["time"][:]
        enspctl16 = cdf_enspctl16.variables[plot_var][:]
        enspctl16_vals = cdf_enspctl16.variables[plot_var][:]
        iunits = cdf_enspctl16[plot_var].units
        enspctl16_vals = -unit_converter(enspctl16_vals, iunits, flux_ounits) * gt2mmSLE

        enspctl84 = cdf_enspctl84.variables[plot_var][:]
        enspctl84_vals = cdf_enspctl84.variables[plot_var][:]
        iunits = cdf_enspctl84[plot_var].units
        enspctl84_vals = -unit_converter(enspctl84_vals, iunits, flux_ounits) * gt2mmSLE

        date = np.arange(start_year + step, start_year + (len(t[:]) + 1), step)

        # ensemble between 16th and 84th quantile
        ax[3].fill_between(
            date[: end_years[k]],
            enspctl16_vals[: end_years[k]],
            enspctl84_vals[: end_years[k]],
            color=rcp_shade_col_dict[rcp],
            linewidth=0,
        )

        ax[3].plot(
            date[: end_years[k]],
            enspctl16_vals[: end_years[k]],
            color=rcp_col_dict[rcp],
            linestyle="solid",
            linewidth=0.2,
        )

        ax[3].plot(
            date[: end_years[k]],
            enspctl84_vals[: end_years[k]],
            color=rcp_col_dict[rcp],
            linestyle="solid",
            linewidth=0.2,
        )

        if ctrl_file is not None:
            print(("Reading RCP {} files".format(rcp)))
            rcp_ctrl_file = [
                f for f in ctrl_file if ("rcp_{}".format(rcp) in f) and not ("tas" in f) and ("flux_absolute" in f)
            ]
            cdf_ctrl = cdo.runmean(runmean_window, input=rcp_ctrl_file, returnCdf=True, options=pthreads)
            ctrl_t = cdf_ctrl.variables["time"][:]
            ctrl_date = np.arange(start_year + step, start_year + (len(ctrl_t[:]) + 1), step)

            ctrl_vals = cdf_ctrl.variables[plot_var][:]
            iunits = cdf_ctrl[plot_var].units
            ctrl_vals = -unit_converter(ctrl_vals, iunits, flux_ounits) * gt2mmSLE

            ax[3].plot(
                ctrl_date[:], ctrl_vals, color=rcp_col_dict[rcp], label=rcp_dict[rcp], linestyle="solid", linewidth=lw
            )

    print("Discharge Contribution Relative")
    plot_var = "discharge_contrib"
    for k, rcp in enumerate(rcp_list[::-1]):

        rcp_files = [f for f in ifiles if ("rcp_{}".format(rcp) in f) and ("flux_percent" in f)]
        pctl16_file = [f for f in rcp_files if "enspctl16" in f][0]
        pctl84_file = [f for f in rcp_files if "enspctl84" in f][0]

        cdf_enspctl16 = cdo.runmean(runmean_window, input=pctl16_file, returnCdf=True, options=pthreads)
        cdf_enspctl84 = cdo.runmean(runmean_window, input=pctl84_file, returnCdf=True, options=pthreads)
        t = cdf_enspctl16.variables["time"][:]

        enspctl16 = cdf_enspctl16.variables[plot_var][:]
        enspctl16_vals = cdf_enspctl16.variables[plot_var][:]

        enspctl84 = cdf_enspctl84.variables[plot_var][:]
        enspctl84_vals = cdf_enspctl84.variables[plot_var][:]
        date = np.arange(start_year + step, start_year + (len(t[:]) + 1), step)

        # ensemble between 16th and 84th quantile
        ax[4].fill_between(
            date[: end_years[k]],
            enspctl16_vals[: end_years[k]],
            enspctl84_vals[: end_years[k]],
            color=rcp_shade_col_dict[rcp],
            linewidth=0,
        )

        ax[4].plot(
            date[: end_years[k]],
            enspctl16_vals[: end_years[k]],
            color=rcp_col_dict[rcp],
            linestyle="solid",
            linewidth=0.2,
        )

        ax[4].plot(
            date[: end_years[k]],
            enspctl84_vals[: end_years[k]],
            color=rcp_col_dict[rcp],
            linestyle="solid",
            linewidth=0.2,
        )

        if ctrl_file is not None:
            rcp_ctrl_file = [f for f in ctrl_file if ("rcp_{}".format(rcp) in f) and ("flux_percent" in f)][0]
            cdf_ctrl = cdo.runmean(runmean_window, input=rcp_ctrl_file, returnCdf=True, options=pthreads)
            ctrl_t = cdf_ctrl.variables["time"][:]
            cdf_date = np.arange(start_year + step, start_year + (len(ctrl_t[:]) + 1), step)

            ctrl_vals = cdf_ctrl.variables[plot_var][:]
            ax[4].plot(cdf_date[:], ctrl_vals, color=rcp_col_dict[rcp], linestyle="solid", linewidth=lw)
    if do_legend:
        legend = ax[3].legend(
            loc="upper right", edgecolor="0", bbox_to_anchor=(0, 0, 0.92, 0.58), bbox_transform=plt.gcf().transFigure
        )
        legend.get_frame().set_linewidth(0.0)
        legend.get_frame().set_alpha(0.0)

    ax[0].set_ylabel("T-anomaly\n(K)")
    ax[1].set_ylabel("$\Delta$(GMSL)\n(m)")
    ax[2].set_ylabel("$\dot M$\n(mm SLE yr$^{\mathregular{-1}}$)")
    ax[3].set_ylabel("$\dot D_{\dot M}$ \n(mm SLE yr$^{\mathregular{-1}}$)")
    ax[4].set_ylabel("$\dot D_{\%}$ (%)")
    ax[4].set_xlabel("Year")

    add_inner_title(ax[0], "A", "upper left")
    add_inner_title(ax[1], "B", "upper left")
    add_inner_title(ax[2], "C", "upper left")
    add_inner_title(ax[3], "D", "upper left")
    add_inner_title(ax[4], "E", "upper left")

    if time_bounds:
        ax[3].set_xlim(time_bounds[0], time_bounds[1])

    if bounds:
        ax[3].set_ylim(bounds[0], bounds[1])

    ymin, ymax = ax[1].get_ylim()

    ax[1].yaxis.set_major_formatter(FormatStrFormatter("%1.0f"))

    if rotate_xticks:
        ticklabels = ax[3].get_xticklabels()
        for tick in ticklabels:
            tick.set_rotation(30)
    else:
        ticklabels = ax[3].get_xticklabels()
        for tick in ticklabels:
            tick.set_rotation(0)

    if title is not None:
        plt.title(title)

    # set_size(2.44, 0.86)

    for out_format in out_formats:
        out_file = outfile + "_rcp_les." + out_format
        print("  - writing image %s ..." % out_file)
        fig.savefig(out_file, bbox_inches="tight", dpi=out_res)


def plot_forcing_mass(plot_var=mass_plot_var):

    fig, ax = plt.subplots(2, 1, sharex="col", sharey="row", figsize=[3, 2])
    fig.subplots_adjust(hspace=0.05, wspace=0.30)

    fig2, ax2 = plt.subplots(2, 1, sharex="col", sharey="row", figsize=[3, 2])
    fig2.subplots_adjust(hspace=0.05, wspace=0.30)

    print("Forcing")
    plot_var = "delta_T"
    for k, rcp in reversed(list(enumerate(rcp_list[::-1]))):

        rcp_files = [f for f in ifiles if "rcp{}".format(rcp) in f]

        ensmin_file = [f for f in rcp_files if ("ENSMIN" in f) and ("tas" in f)][0]
        ensmax_file = [f for f in rcp_files if ("ENSMAX" in f) and ("tas" in f)][0]
        ensmean_file = [f for f in rcp_files if ("ENSMEAN" in f) and ("tas" in f)][0]
        ensmin_cdf = cdo.readCdf(ensmin_file)
        ensmax_cdf = cdo.readCdf(ensmax_file)
        ensmean_cdf = cdo.readCdf(ensmean_file)

        t = ensmin_cdf.variables["time"][:]
        ensmin_date = np.arange(start_year + step, start_year + (len(t[:]) + 1) * step, step)

        t = ensmax_cdf.variables["time"][:]
        ensmax_date = np.arange(start_year + step, start_year + (len(t[:]) + 1) * step, step)

        t = ensmean_cdf.variables["time"][:]
        ensmean_date = np.arange(start_year + step, start_year + (len(t[:]) + 1) * step, step)

        ensmin_vals = np.squeeze(ensmin_cdf.variables[plot_var][:])
        ensmax_vals = np.squeeze(ensmax_cdf.variables[plot_var][:])
        ensmean_vals = np.squeeze(ensmean_cdf.variables[plot_var][:])

        ax[0].fill_between(ensmean_date, ensmin_vals, ensmax_vals, linewidth=0.0, color=rcp_shade_col_dict[rcp])

        ax[0].plot(ensmin_date, ensmin_vals, color=rcp_col_dict[rcp], linewidth=0.20)
        ax[0].plot(ensmax_date, ensmax_vals, color=rcp_col_dict[rcp], linewidth=0.20)
        ax[0].plot(ensmean_date, ensmean_vals, color=rcp_col_dict[rcp], label=rcp_dict[rcp], linewidth=lw)
        ax2[0].plot(ensmean_date, ensmean_vals, color=rcp_col_dict[rcp], label=rcp_dict[rcp], linewidth=lw)

    print("Cumulative Mass")
    plot_var = "limnsw"
    for k, rcp in enumerate(rcp_list[::-1]):

        print(("Reading RCP {} files".format(rcp)))
        rcp_files = [f for f in ifiles if ("rcp_{}".format(rcp) in f) and not ("tas" in f)]

        pctl16_file = [f for f in rcp_files if "enspctl16" in f][0]
        pctl84_file = [f for f in rcp_files if "enspctl84" in f][0]

        cdf_enspctl16 = cdo.readCdf(pctl16_file)
        cdf_enspctl84 = cdo.readCdf(pctl84_file)
        t = cdf_enspctl16.variables["time"][:]

        enspctl16 = cdf_enspctl16.variables[plot_var][:]
        enspctl16_vals = cdf_enspctl16.variables[plot_var][:] - cdf_enspctl16.variables[plot_var][0]
        iunits = cdf_enspctl16[plot_var].units
        enspctl16_vals = -unit_converter(enspctl16_vals, iunits, mass_ounits) * gt2mSLE

        enspctl84 = cdf_enspctl84.variables[plot_var][:]
        enspctl84_vals = cdf_enspctl84.variables[plot_var][:] - cdf_enspctl84.variables[plot_var][0]
        iunits = cdf_enspctl84[plot_var].units
        enspctl84_vals = -unit_converter(enspctl84_vals, iunits, mass_ounits) * gt2mSLE

        date = np.arange(start_year + step, start_year + (len(t[:]) + 1), step)

        # ensemble between 16th and 84th quantile
        ax[1].fill_between(date[:], enspctl16_vals, enspctl84_vals, color=rcp_shade_col_dict[rcp], linewidth=0)

        ax[1].plot(date[:], enspctl16_vals, color=rcp_col_dict[rcp], linestyle="solid", linewidth=0.20)

        ax[1].plot(date[:], enspctl84_vals, color=rcp_col_dict[rcp], linestyle="solid", linewidth=0.20)

        if ctrl_file is not None:
            rcp_ctrl_file = [f for f in ctrl_file if "rcp_{}".format(rcp) in f][0]

            cdf_ctrl = cdo.readCdf(rcp_ctrl_file)
            ctrl_t = cdf_ctrl.variables["time"][:]
            cdf_date = np.arange(start_year + step, start_year + (len(ctrl_t[:]) + 1), step)

            ctrl_vals = cdf_ctrl.variables[plot_var][:] - cdf_ctrl.variables[plot_var][0]
            iunits = cdf_ctrl[plot_var].units
            ctrl_vals = -unit_converter(ctrl_vals, iunits, mass_ounits) * gt2mSLE
            ax[1].plot(cdf_date[:], ctrl_vals, color=rcp_col_dict[rcp], linestyle="solid", linewidth=lw)
            ax2[1].plot(cdf_date[:], ctrl_vals, color=rcp_col_dict[rcp], linestyle="solid", linewidth=lw)

    if do_legend:
        legend = ax[1].legend(
            loc="upper left", edgecolor="0", bbox_to_anchor=(0.0, 0.0, 0, 0), bbox_transform=plt.gcf().transFigure
        )
        legend.get_frame().set_linewidth(0.0)
        legend.get_frame().set_alpha(0.0)
        legend = ax2[0].legend(loc="lower left", edgecolor="0")
        legend.get_frame().set_linewidth(0.0)
        legend.get_frame().set_alpha(0.0)

    ax[0].set_ylabel("T-anomaly\n(K)")
    ax[1].set_ylabel("sea-level\n(m)")
    ax2[0].set_ylabel("T-anomaly\n(K)")
    ax2[1].set_ylabel("sea-level\n(m)")

    add_inner_title(ax[0], "a", "upper left")
    add_inner_title(ax[1], "b", "upper left")
    add_inner_title(ax2[0], "a", "upper left")
    add_inner_title(ax2[1], "b", "upper left")

    if time_bounds:
        ax[1].set_xlim(time_bounds[0], time_bounds[1])
        ax2[1].set_xlim(time_bounds[0], time_bounds[1])

    ax[0].set_ylim(-1, 16)
    ax2[0].set_ylim(-1, 16)
    ax[1].set_ylim(0, 8)
    ax2[1].set_ylim(0, 8)

    ymin, ymax = ax[1].get_ylim()
    ax[1].yaxis.set_major_formatter(FormatStrFormatter("%1.0f"))
    ax2[1].yaxis.set_major_formatter(FormatStrFormatter("%1.0f"))

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

    # set_size(2.44, 0.86)

    for out_format in out_formats:
        out_file = outfile + "_les." + out_format
        print("  - writing image %s ..." % out_file)
        fig.savefig(out_file, bbox_inches="tight", dpi=out_res)
        out_file = outfile + "_ctrl." + out_format
        print("  - writing image %s ..." % out_file)
        fig2.savefig(out_file, bbox_inches="tight", dpi=out_res)


def plot_mass_contrib_d(plot_var=mass_plot_var):

    end_years = [992, 992, 992]

    for k, rcp in enumerate(rcp_list[::-1]):

        fig, ax = plt.subplots(2, 1, sharex="col", sharey="row", figsize=[3, 2])
        fig.subplots_adjust(hspace=0.05, wspace=0.30)

        print("Cumulative Mass")
        plot_var = "limnsw"

        print(("Reading RCP {} files".format(rcp)))
        rcp_files = [f for f in ifiles if ("rcp_{}".format(rcp) in f) and not ("tas" in f)]

        pctl16_file = [f for f in rcp_files if "enspctl16" in f][0]
        pctl84_file = [f for f in rcp_files if "enspctl84" in f][0]

        cdf_enspctl16 = cdo.readCdf(pctl16_file)
        cdf_enspctl84 = cdo.readCdf(pctl84_file)
        t = cdf_enspctl16.variables["time"][:]

        enspctl16 = cdf_enspctl16.variables[plot_var][:]
        enspctl16_vals = cdf_enspctl16.variables[plot_var][:] - cdf_enspctl16.variables[plot_var][0]
        iunits = cdf_enspctl16[plot_var].units
        enspctl16_vals = -unit_converter(enspctl16_vals, iunits, mass_ounits) * gt2mSLE

        enspctl84 = cdf_enspctl84.variables[plot_var][:]
        enspctl84_vals = cdf_enspctl84.variables[plot_var][:] - cdf_enspctl84.variables[plot_var][0]
        iunits = cdf_enspctl84[plot_var].units
        enspctl84_vals = -unit_converter(enspctl84_vals, iunits, mass_ounits) * gt2mSLE

        date = np.arange(start_year + step, start_year + (len(t[:]) + 1), step)

        # ensemble between 16th and 84th quantile
        ax[0].fill_between(date[:], enspctl16_vals, enspctl84_vals, color=rcp_shade_col_dict[rcp], linewidth=0)

        ax[0].plot(date[:], enspctl16_vals, color=rcp_col_dict[rcp], linestyle="solid", linewidth=0.20)

        ax[0].plot(date[:], enspctl84_vals, color=rcp_col_dict[rcp], linestyle="solid", linewidth=0.20)

        if ctrl_file is not None:
            rcp_ctrl_file = [f for f in ctrl_file if "rcp_{}".format(rcp) in f][0]

            cdf_ctrl = cdo.readCdf(rcp_ctrl_file)
            ctrl_t = cdf_ctrl.variables["time"][:]
            cdf_date = np.arange(start_year + step, start_year + (len(ctrl_t[:]) + 1), step)

            ctrl_vals = cdf_ctrl.variables[plot_var][:] - cdf_ctrl.variables[plot_var][0]
            iunits = cdf_ctrl[plot_var].units
            ctrl_vals = -unit_converter(ctrl_vals, iunits, mass_ounits) * gt2mSLE
            ax[0].plot(cdf_date[:], ctrl_vals, color=rcp_col_dict[rcp], linestyle="solid", linewidth=lw)

        print("Discharge Contribution Relative")
        plot_var = "discharge_contrib"

        rcp_files = [f for f in ifiles if ("rcp_{}".format(rcp) in f) and ("flux_percent" in f)]
        pctl16_file = [f for f in rcp_files if "enspctl16" in f][0]
        pctl84_file = [f for f in rcp_files if "enspctl84" in f][0]

        cdf_enspctl16 = cdo.runmean(runmean_window, input=pctl16_file, returnCdf=True, options=pthreads)
        cdf_enspctl84 = cdo.runmean(runmean_window, input=pctl84_file, returnCdf=True, options=pthreads)
        t = cdf_enspctl16.variables["time"][:]

        enspctl16 = cdf_enspctl16.variables[plot_var][:]
        enspctl16_vals = cdf_enspctl16.variables[plot_var][:]

        enspctl84 = cdf_enspctl84.variables[plot_var][:]
        enspctl84_vals = cdf_enspctl84.variables[plot_var][:]
        date = np.arange(start_year + step, start_year + (len(t[:]) + 1), step)

        # ensemble between 16th and 84th quantile
        ax[1].fill_between(
            date[: end_years[k]],
            enspctl16_vals[: end_years[k]],
            enspctl84_vals[: end_years[k]],
            color=rcp_shade_col_dict[rcp],
            linewidth=0,
        )

        ax[1].plot(
            date[: end_years[k]],
            enspctl16_vals[: end_years[k]],
            color=rcp_col_dict[rcp],
            linestyle="solid",
            linewidth=0.2,
        )

        ax[1].plot(
            date[: end_years[k]],
            enspctl84_vals[: end_years[k]],
            color=rcp_col_dict[rcp],
            linestyle="solid",
            linewidth=0.2,
        )

        if ctrl_file is not None:
            rcp_ctrl_file = [f for f in ctrl_file if ("rcp_{}".format(rcp) in f) and ("flux_percent" in f)][0]
            cdf_ctrl = cdo.runmean(runmean_window, input=rcp_ctrl_file, returnCdf=True, options=pthreads)
            ctrl_t = cdf_ctrl.variables["time"][:]
            cdf_date = np.arange(start_year + step, start_year + (len(ctrl_t[:]) + 1), step)

            ctrl_vals = cdf_ctrl.variables[plot_var][:]
            ax[1].plot(cdf_date[:], ctrl_vals, color=rcp_col_dict[rcp], linestyle="solid", linewidth=lw)

        if do_legend:
            legend = ax[1].legend(
                loc="upper left", edgecolor="0", bbox_to_anchor=(0.0, 0.0, 0, 0), bbox_transform=plt.gcf().transFigure
            )
            legend.get_frame().set_linewidth(0.0)
            legend.get_frame().set_alpha(0.0)

        ax[0].set_ylabel("$\Delta$(GMSL)\n(m)")
        ax[1].set_ylabel("$\dot D_{\%}$ (%)")

        if time_bounds:
            ax[1].set_xlim(time_bounds[0], time_bounds[1])

        # ax[0].set_ylim(-1, 16)
        # ax[1].set_ylim(0, 8)

        ymin, ymax = ax[1].get_ylim()
        ax[1].yaxis.set_major_formatter(FormatStrFormatter("%1.0f"))

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

        # set_size(2.44, 0.86)

        for out_format in out_formats:
            out_file = outfile + "_mass_d_rcp_{}.".format(rcp) + out_format
            print("  - writing image %s ..." % out_file)
            fig.savefig(out_file, bbox_inches="tight", dpi=out_res)


def plot_sobel(plot_var=mass_plot_var):

    end_years = [992, 992, 992]

    categories = ["Climate", "Surface", "Ocean", "Ice Dynamics"]
    category_col_dict = {"Climate": "#81c77f", "Surface": "#886c62", "Ocean": "#beaed4", "Ice Dynamics": "#dcd588"}

    fig, ax = plt.subplots(len(rcp_list), 1, sharex="col", sharey="row", figsize=[3, 2])
    fig.subplots_adjust(hspace=0.2, wspace=0.30)

    for k, rcp in enumerate(rcp_list):

        years = range(int(time_bounds[0]), int(time_bounds[-1] + 1))

        nt = len(years)
        nc = len(categories)
        climate = np.zeros(nt)
        surface = np.zeros(nt)
        ocean = np.zeros(nt)
        ice = np.zeros(nt)
        for t, year in enumerate(years):
            filename = input_filename(prefix, rcp, year)
            mdata = read_sobel_file(filename) * 100  # convert to percent
            climate[t] = mdata[0] + mdata[3]
            surface[t] = mdata[1] + mdata[2] + mdata[4]
            ocean[t] = mdata[5] + mdata[6] + mdata[7] + mdata[8]
            ice[t] = mdata[9] + mdata[10]

        ax[k].fill_between(years, np.zeros(nt), climate, color=category_col_dict[categories[0]], label=categories[0])
        ax[k].fill_between(
            years, climate, climate + surface, color=category_col_dict[categories[1]], label=categories[1]
        )
        ax[k].fill_between(
            years,
            climate + surface,
            climate + surface + ocean,
            color=category_col_dict[categories[2]],
            label=categories[2],
        )
        ax[k].fill_between(
            years,
            climate + surface + ocean,
            climate + surface + ocean + ice,
            color=category_col_dict[categories[3]],
            label=categories[3],
        )

        ax[k].set_ylim(0, 100)
        if time_bounds:
            ax[k].set_xlim(time_bounds[0], time_bounds[1])

        ymin, ymax = ax[1].get_ylim()
        ax[k].yaxis.set_major_formatter(FormatStrFormatter("%1.0f"))
        add_inner_title(ax[k], "{}".format(rcp_dict[rcp]), "upper left")

    ax[1].set_ylabel("Variance (%)")
    ax[-1].set_xlabel("Year")
    if do_legend:
        legend = ax[-1].legend(
            ncol=4,
            loc="upper left",
            edgecolor="0",
            bbox_to_anchor=(0.0, 0.0, 0, 0),
            bbox_transform=plt.gcf().transFigure,
        )
        legend.get_frame().set_linewidth(0.0)
        legend.get_frame().set_alpha(0.0)

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
        out_file = outfile + "_ts." + out_format
        print("  - writing image %s ..." % out_file)
        fig.savefig(out_file, bbox_inches="tight", dpi=out_res)


def plot_random_flux(plot_var=flux_plot_var):

    fig = plt.figure()
    offset = transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)
    ax = fig.add_subplot(111)

    fnos = np.random.randint(0, len(ifiles), size=25)

    for fno in fnos:

        ifile = ifiles[fno]
        cdf = cdo.runmean(runmean_window, input=ifile, returnCdf=True, options=pthreads)
        t = cdf.variables["time"][:]

        vals = -cdf.variables[plot_var][:]

        date = np.arange(start_year + step, start_year + (len(t[:]) + 1), step)

        ax.plot(date, vals, linestyle="solid", linewidth=0.25)

    set_size(2.44, 0.86)

    for out_format in out_formats:
        out_file = outfile + "_random" + "_" + plot_var + "." + out_format
        print("  - writing image %s ..." % out_file)
        fig.savefig(out_file, bbox_inches="tight", dpi=out_res)


def plot_flux_partitioning():

    fig, axa = plt.subplots(4, 3, sharex="col", sharey="row", figsize=[6, 4])
    fig.subplots_adjust(hspace=0.05, wspace=0.05)

    for k, rcp in enumerate(rcp_list):
        if rcp == "26":
            m = 0
        elif rcp == "45":
            m = 1
        else:
            m = 2

        alpha = 0.25
        rcp_ctrl_file = [f for f in ifiles if "rcp_{}".format(rcp) in f and "CTRL" in f][0]
        rcp_ntrl_file = [f for f in ifiles if "rcp_{}".format(rcp) in f and "NTRL" in f in f][0]

        cdf = cdo.runmean(runmean_window, input=rcp_ctrl_file, returnCdf=True, options=pthreads)
        cdf_ntrl = cdo.runmean(runmean_window, input=rcp_ntrl_file, returnCdf=True, options=pthreads)
        t = cdf.variables["time"][:]
        date = np.arange(start_year + step, start_year + (len(t[:]) + 1), step)

        area_var = "ice_area_glacierized"
        area_vals = cdf.variables[area_var][:]
        area_iunits = cdf[area_var].units

        mask = 1 - (area_vals[0] - area_vals) / area_vals[0] < 0.10

        tom_var = "dMdt"
        tom_vals = np.squeeze(cdf.variables[tom_var][:])
        tom_s_vals = tom_vals / area_vals
        tom_iunits = cdf[tom_var].units
        tom_vals = unit_converter(tom_vals, tom_iunits, flux_ounits)
        tom_s_iunits = cf_units.Unit(tom_iunits) / cf_units.Unit(area_iunits)
        tom_s_vals = tom_s_iunits.convert(tom_s_vals, specific_flux_ounits)
        tom_s_ma_vals = np.ma.array(tom_s_vals, mask=mask)

        snow_var = "surface_accumulation_rate"
        snow_vals = np.squeeze(cdf.variables[snow_var][:])
        snow_s_vals = snow_vals / area_vals
        snow_iunits = cdf[snow_var].units
        snow_vals = unit_converter(snow_vals, snow_iunits, flux_ounits)
        snow_s_iunits = cf_units.Unit(snow_iunits) / cf_units.Unit(area_iunits)
        snow_s_vals = snow_s_iunits.convert(snow_s_vals, specific_flux_ounits)
        snow_s_ma_vals = np.ma.array(snow_s_vals, mask=mask)

        ru_var = "surface_runoff_rate"
        ru_vals = np.squeeze(cdf.variables[ru_var][:])
        ru_ntrl_vals = np.squeeze(cdf_ntrl.variables[ru_var][:])
        ru_s_vals = ru_vals / area_vals
        ru_ntrl_s_vals = ru_ntrl_vals / area_vals
        ru_iunits = cdf[ru_var].units
        ru_vals = -unit_converter(ru_vals, ru_iunits, flux_ounits)
        ru_ntrl_iunits = cdf_ntrl[ru_var].units
        ru_ntrl_vals = -unit_converter(ru_ntrl_vals, ru_iunits, flux_ounits)
        ru_s_iunits = cf_units.Unit(ru_iunits) / cf_units.Unit(area_iunits)
        ru_s_vals = -ru_s_iunits.convert(ru_s_vals, specific_flux_ounits)
        ru_s_ma_vals = np.ma.array(ru_s_vals, mask=mask)
        ru_ntrl_s_vals = -ru_s_iunits.convert(ru_ntrl_s_vals, specific_flux_ounits)
        ru_ntrl_s_ma_vals = np.ma.array(ru_ntrl_s_vals, mask=mask)

        d_var = "tendency_of_ice_mass_due_to_discharge"
        d_vals = np.squeeze(cdf.variables[d_var][:])
        d_s_vals = d_vals / area_vals
        d_iunits = cdf[d_var].units
        d_vals = unit_converter(d_vals, d_iunits, flux_ounits)
        d_s_iunits = cf_units.Unit(d_iunits) / cf_units.Unit(area_iunits)
        d_s_vals = d_s_iunits.convert(d_s_vals, specific_flux_ounits)
        d_s_ma_vals = np.ma.array(d_s_vals, mask=mask)

        b_var = "tendency_of_ice_mass_due_to_basal_mass_flux"
        b_vals = np.squeeze(cdf.variables[b_var][:])
        b_s_vals = b_vals / area_vals
        b_iunits = cdf[b_var].units
        b_vals = unit_converter(b_vals, b_iunits, flux_ounits)
        b_s_iunits = cf_units.Unit(b_iunits) / cf_units.Unit(area_iunits)
        b_s_vals = b_s_iunits.convert(b_s_vals, specific_flux_ounits)
        b_s_ma_vals = np.ma.array(b_s_vals, mask=mask)

        la, = axa[0, m].plot(date, area_vals / 1e12, color="#084594", label="area")
        axa[0, m].set_aspect(200, anchor="S", adjustable="box-forced")
        axa[0, m].set_title("{}".format(rcp_dict[rcp]))

        # Don't plot basal mass balance
        b_vals = b_s_vals = 0

        axa[1, m].fill_between(date, 0, snow_vals, color="#6baed6", label="accumulation", linewidth=0, alpha=alpha)
        lsn = axa[1, m].fill_between(date, 0, snow_vals, color="#6baed6", label="accumulation", linewidth=0)
        lruw = axa[1, m].fill_between(
            date, b_vals, b_vals + ru_vals, color="#fb6a4a", label="runoff (elevation)", linewidth=0
        )
        lrul = axa[1, m].fill_between(
            date, b_vals, b_vals + ru_ntrl_vals, color="#fdae6b", label="runoff (climate)", linewidth=0
        )
        ld = axa[1, m].fill_between(
            date,
            b_vals + np.minimum(ru_vals, ru_ntrl_vals),
            b_vals + np.minimum(ru_vals, ru_ntrl_vals) + d_vals,
            color="#74c476",
            label="discharge",
            linewidth=0,
        )

        axa[1, m].axhline(0, color="k", linestyle="dotted")
        axa[1, m].plot(date, snow_vals, color="#2171b5", linewidth=0.3)
        axa[1, m].plot(date, b_vals + ru_vals, color="#e6550d", linewidth=0.3)
        axa[1, m].plot(date, b_vals + ru_ntrl_vals, color="#cb181d", linewidth=0.3)
        axa[1, m].plot(date, b_vals + np.minimum(ru_vals, ru_ntrl_vals) + d_vals, color="#238b45", linewidth=0.3)
        lmb, = axa[1, m].plot(date, tom_vals, color="k", label="mass balance", linewidth=0.6)

        axa[2, m].fill_between(date, 0, snow_s_vals, color="#6baed6", label="accumulation", linewidth=0, alpha=alpha)
        lsn = axa[2, m].fill_between(date, 0, snow_s_ma_vals, color="#6baed6", label="accumulation", linewidth=0)
        axa[2, m].fill_between(
            date, b_s_vals, b_s_vals + ru_s_vals, color="#fb6a4a", label="runoff (elevation)", linewidth=0, alpha=alpha
        )
        lruw = axa[2, m].fill_between(
            date, b_s_vals, b_s_ma_vals + ru_s_ma_vals, color="#fb6a4a", label="runoff (elevation)", linewidth=0
        )
        axa[2, m].fill_between(
            date,
            b_s_vals,
            b_s_vals + ru_ntrl_s_vals,
            color="#fdae6b",
            label="runoff (climate)",
            linewidth=0,
            alpha=alpha,
        )
        lrul = axa[2, m].fill_between(
            date, b_s_ma_vals, b_s_ma_vals + ru_ntrl_s_ma_vals, color="#fdae6b", label="runoff (climate)", linewidth=0
        )
        axa[2, m].fill_between(
            date,
            b_s_vals + np.minimum(ru_s_vals, ru_ntrl_s_vals),
            b_s_vals + np.minimum(ru_s_vals, ru_ntrl_s_vals) + d_s_vals,
            color="#74c476",
            label="discharge",
            linewidth=0,
            alpha=alpha,
        )
        ld = axa[2, m].fill_between(
            date,
            b_s_ma_vals + np.minimum(ru_s_ma_vals, ru_ntrl_s_ma_vals),
            b_s_ma_vals + np.minimum(ru_s_ma_vals, ru_ntrl_s_ma_vals) + d_s_ma_vals,
            color="#74c476",
            label="discharge",
            linewidth=0,
        )

        axa[2, m].axhline(0, color="k", linestyle="dotted")
        axa[2, m].plot(date, snow_s_vals, color="#2171b5", linewidth=0.3)
        axa[2, m].plot(date, b_s_vals + ru_ntrl_s_vals, color="#e6550d", linewidth=0.3)
        axa[2, m].plot(date, b_s_vals + ru_s_vals, color="#cb181d", linewidth=0.3)
        axa[2, m].plot(
            date, b_s_vals + np.minimum(ru_s_vals, ru_ntrl_s_vals) + d_s_vals, color="#238b45", linewidth=0.3
        )
        lmb, = axa[2, m].plot(date, tom_s_vals, color="k", label="mass balance", linewidth=0.6)

        axa[3, m].axhline(100, color="k", linestyle="dotted")
        axa[3, m].plot(date, -ru_vals / snow_vals * 100, color="#cb181d", label="runoff (total)", linewidth=0.4)
        axa[3, m].plot(date, -d_vals / snow_vals * 100, color="#238b45", label="discharge", linewidth=0.4)
        axa[3, m].plot(date, -tom_vals / snow_vals * 100, color="#000000", label="mass balance", linewidth=0.6)

        axa[3, m].set_xlabel("Year")

    axa[0, 0].set_ylabel("Area\n(10$^{6}$ km$^{\mathregular{2}}$)")
    axa[1, 0].set_ylabel("Rate\n(Gt yr$^{\mathregular{-1}}$)")
    axa[2, 0].set_ylabel("Rate\n(kg m$^{\mathregular{-2}}$ yr$^{\mathregular{-1}}$)")
    axa[3, 0].set_ylabel("Ratio\n(%)")

    axa[2, 0].set_ylim(-12000, 4000)
    axm = axa[2, 2].twinx()
    ymi, yma = axa[2, 2].get_ylim()
    axm.set_ylim(ymi / 910.0, yma / 910.0)
    axm.set_yticks([-10, -6, -4, -2, 0, 2, 4])
    axm.set_ylabel("(m yr$^{\mathregular{-1}}$ ice equiv.)")

    legend = axa[0, 0].legend(
        handles=[la],
        loc="lower left",
        ncol=1,
        labelspacing=0.01,
        handlelength=1.5,
        columnspacing=1,
        edgecolor="0",
        bbox_to_anchor=(0.205, 0.72, 0, 0),
        bbox_transform=plt.gcf().transFigure,
    )
    legend.get_frame().set_linewidth(0.0)
    legend.get_frame().set_alpha(0.0)

    legend = axa[2, 0].legend(
        handles=[lsn, lrul, lruw, ld, lmb],
        loc="lower left",
        ncol=1,
        labelspacing=0.08,
        handlelength=1.5,
        columnspacing=1,
        edgecolor="0",
        bbox_to_anchor=(0.205, 0.48, 0, 0),
        bbox_transform=plt.gcf().transFigure,
    )
    legend.get_frame().set_linewidth(0.0)
    legend.get_frame().set_alpha(0.0)

    legend = axa[3, 0].legend(
        loc="lower left",
        ncol=1,
        labelspacing=0.1,
        handlelength=1.5,
        columnspacing=1,
        edgecolor="0",
        bbox_to_anchor=(0.205, 0.18, 0, 0),
        bbox_transform=plt.gcf().transFigure,
    )
    legend.get_frame().set_linewidth(0.0)
    legend.get_frame().set_alpha(0.0)

    if time_bounds:
        for o in range(0, 3):
            for p in range(0, 3):
                axa[o, p].set_xlim(time_bounds[0], time_bounds[1])

    # ax.yaxis.set_major_formatter(FormatStrFormatter('%1.0f'))

    add_inner_title(axa[0, 0], "A", "lower left")
    add_inner_title(axa[0, 1], "B", "lower left")
    add_inner_title(axa[0, 2], "C", "lower left")
    add_inner_title(axa[1, 0], "D", "lower left")
    add_inner_title(axa[1, 1], "E", "lower left")
    add_inner_title(axa[1, 2], "F", "lower left")
    add_inner_title(axa[2, 0], "G", "lower left")
    add_inner_title(axa[2, 1], "H", "lower left")
    add_inner_title(axa[2, 2], "I", "lower left")
    add_inner_title(axa[3, 0], "J", "upper left")
    add_inner_title(axa[3, 1], "K", "upper left")
    add_inner_title(axa[3, 2], "L", "upper left")

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
        out_file = outfile + "_partitioning." + out_format
        print("  - writing image %s ..." % out_file)
        fig.savefig(out_file, bbox_inches="tight", dpi=out_res)


def plot_basin_flux_partitioning():

    fig, axa = plt.subplots(6, 3, sharex="col", sharey="row", figsize=[6, 4])
    fig.subplots_adjust(hspace=0.06, wspace=0.04)

    for k, rcp in enumerate(rcp_list):
        if rcp == "26":
            m = 0
        elif rcp == "45":
            m = 1
        else:
            m = 2

        for k, basin in enumerate(basin_list):

            basin_files = [f for f in ifiles if "b_{}".format(basin) in f]

            rcp_ctrl_file = [f for f in basin_files if "rcp_{}".format(rcp) in f and "CTRL" in f][0]
            # rcp_ntrl_file = [f for f in basin_files if 'rcp_{}'.format(rcp) in f and 'NTRL' in f in f][0]
            print(("Reading {}".format(rcp_ctrl_file)))
            cdf = cdo.runmean(runmean_window, input=rcp_ctrl_file, returnCdf=True, options=pthreads)

            t = cdf.variables["time"][:]
            date = np.arange(start_year + step, start_year + (len(t[:]) + 1), step)

            tom_var = "dMdt"
            tom_vals = np.squeeze(cdf.variables[tom_var][:])
            tom_iunits = cdf[tom_var].units
            tom_vals = unit_converter(tom_vals, tom_iunits, flux_ounits)

            snow_var = "surface_accumulation_rate"
            snow_vals = np.squeeze(cdf.variables[snow_var][:])
            snow_iunits = cdf[snow_var].units
            snow_vals = unit_converter(snow_vals, snow_iunits, flux_ounits)

            ru_var = "surface_runoff_rate"
            ru_vals = np.squeeze(cdf.variables[ru_var][:])
            # ru_ntrl_vals = np.squeeze(cdf_ntrl.variables[ru_var][:])
            ru_iunits = cdf[ru_var].units
            ru_vals = -unit_converter(ru_vals, ru_iunits, flux_ounits)
            # ru_ntrl_iunits = cdf_ntrl[ru_var].units
            # ru_ntrl_vals = -unit_converter(ru_ntrl_vals, ru_iunits, flux_ounits)

            d_var = "tendency_of_ice_mass_due_to_discharge"
            d_vals = np.squeeze(cdf.variables[d_var][:])
            d_iunits = cdf[d_var].units
            d_vals = unit_converter(d_vals, d_iunits, flux_ounits)

            lsn = axa[k, m].fill_between(date, 0, snow_vals, color="#6baed6", label="accumulation", linewidth=0)
            lruw = axa[k, m].fill_between(date, 0, ru_vals, color="#fb6a4a", label="runoff", linewidth=0)
            # lrul = axa[k,m].fill_between(date, 0, ru_ntrl_vals, color='#fdae6b', label='RW', linewidth=0)
            ld = axa[k, m].fill_between(
                date, ru_vals, ru_vals + d_vals, color="#74c476", label="discharge", linewidth=0
            )
            axa[k, m].plot(date, snow_vals, color="#2171b5", linewidth=0.3)
            axa[k, m].plot(date, ru_vals, color="#cb181d", linewidth=0.3)
            # axa[k,m].plot(date, ru_ntrl_vals, color='#e6550d', linewidth=0.3)
            axa[k, m].plot(date, ru_vals + d_vals, color="#238b45", linewidth=0.3)
            lmb, = axa[k, m].plot(date, tom_vals, color="k", label="mass balance", linewidth=0.6)
            axa[k, m].axhline(0, color="k", linestyle="dotted")

            axa[k, m].yaxis.set_major_formatter(FormatStrFormatter("%1.0f"))

            if k == 5:
                axa[k, m].set_xlabel("Year")
            if m == 0:
                axa[k, m].set_ylabel("Rate\n (Gt yr$^{\mathregular{-1}}$)")

            if time_bounds:
                axa[k, m].set_xlim(time_bounds[0], time_bounds[1])

            if bounds:
                axa[k, m].set_ylim(bounds[0], bounds[1])

            if rotate_xticks:
                ticklabels = axa[k, m].get_xticklabels()
                for tick in ticklabels:
                    tick.set_rotation(30)
            else:
                ticklabels = axa[k, m].get_xticklabels()
                for tick in ticklabels:
                    tick.set_rotation(0)

    legend = axa[0, 2].legend(
        handles=[lsn, lruw, ld, lmb],
        loc="upper right",
        ncol=1,
        labelspacing=0.1,
        handlelength=1.5,
        columnspacing=1,
        edgecolor="0",
        bbox_to_anchor=(0.45, 0.075, 0, 0),
        bbox_transform=plt.gcf().transFigure,
    )
    legend.get_frame().set_linewidth(0.0)
    legend.get_frame().set_alpha(0.0)

    for out_format in out_formats:
        out_file = outfile + "_basin_partitioning." + out_format
        print("  - writing image %s ..." % out_file)
        fig.savefig(out_file, bbox_inches="tight", dpi=out_res)


def plot_ens_mass(plot_var="limnsw"):

    fig = plt.figure()
    offset = transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)
    ax = fig.add_subplot(111)
    for k, rcp in enumerate(rcp_list[::-1]):

        print(("Reading RCP {} files".format(rcp)))
        rcp_files = [f for f in ifiles if "rcp_{}".format(rcp) in f]

        pctl16_file = [f for f in rcp_files if "enspctl16" in f][0]
        pctl84_file = [f for f in rcp_files if "enspctl84" in f][0]

        cdf_enspctl16 = cdo.readCdf(pctl16_file)
        cdf_enspctl84 = cdo.readCdf(pctl84_file)
        t = cdf_enspctl16.variables["time"][:]

        enspctl16 = cdf_enspctl16.variables[plot_var][:]
        enspctl16_vals = cdf_enspctl16.variables[plot_var][:] - cdf_enspctl16.variables[plot_var][0]
        iunits = cdf_enspctl16[plot_var].units
        enspctl16_vals = -unit_converter(enspctl16_vals, iunits, mass_ounits) * gt2mSLE

        enspctl84 = cdf_enspctl84.variables[plot_var][:]
        enspctl84_vals = cdf_enspctl84.variables[plot_var][:] - cdf_enspctl84.variables[plot_var][0]
        iunits = cdf_enspctl84[plot_var].units
        enspctl84_vals = -unit_converter(enspctl84_vals, iunits, mass_ounits) * gt2mSLE

        date = np.arange(start_year + step, start_year + (len(t[:]) + 1), step)

        # ensemble between 16th and 84th quantile
        ax.fill_between(date[:], enspctl16_vals, enspctl84_vals, color=rcp_col_dict[rcp], alpha=0.4, linewidth=0)

        ax.plot(date[:], enspctl16_vals, color=rcp_col_dict[rcp], linestyle="solid", linewidth=0.4)

        ax.plot(date[:], enspctl84_vals, color=rcp_col_dict[rcp], linestyle="solid", linewidth=0.4)

        if ctrl_file is not None:
            rcp_ctrl_file = [f for f in ctrl_file if "rcp_{}".format(rcp) in f][0]

            cdf_ctrl = cdo.readCdf(rcp_ctrl_file)
            ctrl_t = cdf_ctrl.variables["time"][:]
            cdf_date = np.arange(start_year + step, start_year + (len(ctrl_t[:]) + 1), step)

            ctrl_vals = cdf_ctrl.variables[plot_var][:] - cdf_ctrl.variables[plot_var][0]
            iunits = cdf_ctrl[plot_var].units
            ctrl_vals = -unit_converter(ctrl_vals, iunits, mass_ounits) * gt2mSLE
            ax.plot(cdf_date[:], ctrl_vals, color=rcp_col_dict[rcp], linestyle="solid", linewidth=lw)

    if do_legend:
        legend = ax.legend(
            loc="upper right", edgecolor="0", bbox_to_anchor=(0, 0, 0.35, 0.88), bbox_transform=plt.gcf().transFigure
        )
        legend.get_frame().set_linewidth(0.0)
        legend.get_frame().set_alpha(0.0)

    ax.set_xlabel("Year")
    ax.set_ylabel("$\Delta$(GMSL) (m)")

    if time_bounds:
        ax.set_xlim(time_bounds[0], time_bounds[1])

    if bounds:
        ax.set_ylim(bounds[0], bounds[1])

    ymin, ymax = ax.get_ylim()

    ax.yaxis.set_major_formatter(FormatStrFormatter("%1.0f"))

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
        out_file = outfile + "_rcp" + "_" + plot_var + "." + out_format
        print("  - writing image %s ..." % out_file)
        fig.savefig(out_file, bbox_inches="tight", dpi=out_res)


def plot_ctrl_mass(plot_var="limnsw"):

    fig = plt.figure()
    offset = transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)
    ax = fig.add_subplot(111)

    sle = 7.21
    ax.axhline(sle, color="k", linestyle="dashed", linewidth=0.2)

    for k, rcp in enumerate(rcp_list[::-1]):
        rcp_file = [f for f in ifiles if "rcp_{}".format(rcp) in f][0]
        cdf = cdo.readCdf(rcp_file)
        t = cdf.variables["time"][:]
        date = np.arange(start_year + step, start_year + (len(t[:]) + 1), step)
        var_vals = cdf.variables[plot_var][:] - cdf.variables[plot_var][0]
        iunits = cdf.variables[plot_var].units
        var_vals = -unit_converter(var_vals, iunits, mass_ounits) * gt2mSLE

        plt.plot(
            date[var_vals < sle], var_vals[var_vals < sle], color=rcp_col_dict[rcp], linewidth=lw, label=rcp_dict[rcp]
        )

    if do_legend:
        legend = ax.legend(
            loc="center right", edgecolor="0", bbox_to_anchor=(0.91, 0.63), bbox_transform=plt.gcf().transFigure
        )
        legend.get_frame().set_linewidth(0.0)
        legend.get_frame().set_alpha(0.0)

    ax.set_xlabel("Year")
    ax.set_ylabel("$\Delta$(GMSL) (m)")

    if time_bounds:
        ax.set_xlim(time_bounds[0], time_bounds[1])

    if bounds:
        ax.set_ylim(bounds[0], bounds[1])

    ymin, ymax = ax.get_ylim()

    ax.yaxis.set_major_formatter(FormatStrFormatter("%1.2f"))

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
        out_file = outfile + "_ctrl" + "_" + plot_var + "." + out_format
        print("  - writing image %s ..." % out_file)
        fig.savefig(out_file, bbox_inches="tight", dpi=out_res)


def plot_pdfs():

    years = [2100, 2200, 2300]
    ranges_max = [55, 250, 450]
    range_min = -1
    for year, range_max in zip(years, ranges_max):

        nbins = range_max - range_min

        fig = plt.figure()
        ax = fig.add_subplot(111)

        title = "Year {}".format(year)

        for rcp in rcp_list:
            sle = np.loadtxt("les_gcm_rcp{rcp}_{year}.csv".format(year=year, rcp=rcp), delimiter=",")[:, 1]
            ax.hist(
                sle, bins=nbins, range=[range_min, range_max], color=rcp_col_dict[rcp], label=rcp_dict[rcp], alpha=0.5
            )

        if do_legend:
            legend = ax.legend(
                loc="center right", edgecolor="0", bbox_to_anchor=(0.91, 0.63), bbox_transform=plt.gcf().transFigure
            )
            legend.get_frame().set_linewidth(0.0)
            legend.get_frame().set_alpha(0.0)

        ax.set_ylabel("Density")
        ax.set_xlabel("$\Delta$(GMSL) (m)")
        plt.title(title)

        for out_format in out_formats:
            out_file = outfile + "_pdf_" + str(year) + "." + out_format
            print("  - writing image %s ..." % out_file)
            fig.savefig(out_file, bbox_inches="tight", dpi=out_res)


if plot == "les":
    plot_les()
elif plot == "forcing_mass":
    plot_forcing_mass(plot_var="limnsw")
elif plot == "mass_d":
    plot_mass_contrib_d(plot_var="limnsw")
elif plot == "ens_mass":
    plot_ens_mass(plot_var="limnsw")
elif plot == "ctrl_mass":
    plot_ctrl_mass(plot_var="limnsw")
elif plot == "ctrl_mass_anim":
    plot_ctrl_mass_anim(plot_var="limnsw")
elif plot == "d_contrib_anim":
    plot_d_contrib_anim(plot_var="discharge_contrib")
elif plot == "flux_partitioning":
    plot_flux_partitioning()
elif plot == "basin_flux_partitioning":
    plot_basin_flux_partitioning()
elif plot == "cmip5_rcp":
    plot_cmip5_rcp()
elif plot == "grid_res":
    plot_grid_res()
elif plot == "random_flux":
    plot_random_flux(plot_var="tendency_of_ice_mass_due_to_discharge")
elif plot == "pdfs":
    plot_pdfs()
elif plot == "profile":
    plot_profile_ts_combined()
elif plot == "profile_anim":
    plot_profile_ts_animation()
elif plot == "sobel":
    plot_sobel()
