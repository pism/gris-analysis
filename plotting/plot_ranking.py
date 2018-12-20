#!/usr/bin/env python

# Copyright (C) 18 Andy Aschwanden

from argparse import ArgumentParser
import re
import matplotlib.transforms as transforms
from matplotlib.ticker import FormatStrFormatter
import os

import numpy as np
import pandas as pd
import pylab as plt

from unidecode import unidecode

rcp_col_dict = {"CTRL": "k", "85": "#990002", "45": "#5492CD", "26": "#003466"}
rcp_shade_col_dict = {"CTRL": "k", "85": "#F4A582", "45": "#92C5DE", "26": "#4393C3"}
rcp_dict = {"26": "RCP 2.6", "45": "RCP 4.5", "85": "RCP 8.5", "CTRL": "CTRL"}
rcp_maker_dict = {"85": "d", "45": "s", "26": "o"}

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
parser.add_argument("--ranking_file", help="File with Peclet Number rankin g", default=None)

parser.add_argument("--title", dest="title", help="""Plot title.""", default=None)

options = parser.parse_args()
ranking_file = options.ranking_file
mass_files = options.FILE
outfile = options.outfile
out_res = options.out_res
out_formats = options.out_formats
title = options.title

fig = plt.figure()
ax = fig.add_subplot(111)

results = {}
df = pd.read_csv(ranking_file)
for k, mfile in enumerate(mass_files):
    rcp = re.search("rcp(.+?)_", mfile).group(1)
    year = re.search("_(.+?).csv", mfile).group(1)
    print(rcp, year)
    mdf = pd.read_csv(mfile)
    results[k] = {"year": year}
    results[k] = {"rcp": rcp}
    results[k] = {"data": pd.merge(df, mdf, left_on="id_rignot", right_on="# id", sort=True)}
    ax.plot(
        results[k]["data"]["rank"].values,
        results[k]["data"]["dgmsl"].values,
        color=rcp_col_dict[rcp],
        linestyle="None",
        marker=rcp_maker_dict[rcp],
        alpha=0.5,
    )

ax.set_xlabel("Rank")
ax.set_ylabel("Discharge cumulative (Gt)")

if title is not None:
    plt.title(title)

for out_format in out_formats:
    out_file = outfile + "_ranking." + out_format
    print("  - writing image %s ..." % out_file)
    fig.savefig(out_file, bbox_inches="tight", dpi=out_res)
