#!/usr/bin/env python

# Copyright (C) 2018-2019 Andy Aschwanden

from argparse import ArgumentParser
import re
from matplotlib.ticker import FormatStrFormatter
import os

import numpy as np
import pandas as pd
import pylab as plt
import statsmodels.api as sm
from unidecode import unidecode

rcp_col_dict = {"CTRL": "k", "85": "#990002", "45": "#5492CD", "26": "#003466"}
rcp_shade_col_dict = {"CTRL": "k", "85": "#F4A582", "45": "#92C5DE", "26": "#4393C3"}
rcp_dict = {"26": "RCP 2.6", "45": "RCP 4.5", "85": "RCP 8.5", "CTRL": "CTRL"}
marker_dict = {"2100": "d", "2200": "s", "2300": "o"}

# Set up the option parser
parser = ArgumentParser()
parser.description = "A script for PISM output files to time series plots using pylab/matplotlib."
parser.add_argument("FILE", nargs="*")
parser.add_argument(
    "-f",
    "--output_format",
    dest="out_formats",
    help="Comma-separated list with output graphics suffix, default = pdf",
    default=["pdf"],
)
parser.add_argument("--no_legend", dest="do_legend", action="store_false", help="Do not plot legend", default=True)
parser.add_argument(
    "-o",
    "--output_file",
    dest="outfile",
    help="output file name without suffix, i.e. ts_control -> ts_control_variable",
    default="foo",
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
parser.add_argument("--ranking_file", help="File with Peclet Number rankin", default=None)
parser.add_argument("--correlation_file", help="File with correlation", default=None)

parser.add_argument("--title", dest="title", help="""Plot title.""", default=None)

options = parser.parse_args()
correlation_file = options.correlation_file
ranking_file = options.ranking_file
mass_files = options.FILE
outfile = options.outfile
out_res = options.out_res
out_formats = options.out_formats
title = options.title
area_loss_threshold = 0.01
# correlation from Aschwanden, Fahnestock, Truffer (2016)
correlation_threshold = 0.85

fig = plt.figure()
ax = fig.add_subplot(111)

results = {}
df = pd.read_csv(ranking_file)
dfc = pd.read_csv(correlation_file)
df = pd.merge(df, dfc, left_on="id_rignot", right_on="# id", sort=True)
for k, mfile in enumerate(mass_files):
    # Extract RCP from file name
    rcp = re.search("rcp(.+?)_", mfile).group(1)
    # Extract Year from file name
    year = re.search("_(.+?).csv", mfile).group(1)
    # Read PISM area and dynamic mass loss from file
    mdf = pd.read_csv(mfile)
    mdf = pd.merge(df, mdf, left_on="id_rignot", right_on="# id", sort=True)
    # Select glaciers where correlation greater or equal than threshold
    mdf["sftgif"] = pd.to_numeric(mdf["sftgif"])
    mdf = mdf[(mdf["correlation"] >= correlation_threshold) & (mdf["sftgif"] > area_loss_threshold)]
    regression = sm.OLS(mdf["dgmsl"], sm.add_constant(mdf["rank"])).fit()
    bias, trend = regression.params
    results[k] = {"year": year, "rcp": rcp, "data": mdf, "regression": regression}
    outfile = "r_{}_a_{}_rcp_{}_{}.csv".format(correlation_threshold, area_loss_threshold, rcp, year)
    mdf.to_csv(outfile)
    # Make plot
    # ax.plot(
    #     results[k]["data"]["rank"].values,
    #     bias + trend * results[k]["data"]["rank"].values,
    #     color=rcp_col_dict[rcp],
    #     linewidth=0.2,
    # )
    ax.plot(
        results[k]["data"]["rank"].values,
        results[k]["data"]["dgmsl"].values,
        color=rcp_col_dict[rcp],
        linestyle="None",
        marker=marker_dict[year],
        alpha=0.5,
        label="{}, Year {}".format(rcp_dict[rcp], year),
    )

ax.set_xlabel("Rank")
ax.set_ylabel("Discharge cumulative (Gt)")

legend = ax.legend(edgecolor="0")
legend.get_frame().set_linewidth(0.0)
legend.get_frame().set_alpha(0.0)


if title is not None:
    plt.title(title)

for out_format in out_formats:
    out_file = outfile + "_ranking." + out_format
    print("  - writing image %s ..." % out_file)
    fig.savefig(out_file, bbox_inches="tight", dpi=out_res)
