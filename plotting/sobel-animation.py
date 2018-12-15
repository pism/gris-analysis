#!/usr/bin/env python

# Copyright (C) 2018 Andy Aschwanden

from argparse import ArgumentParser
import numpy as np
import pylab as plt

def input_filename(prefix, rcp, year):
    return "{prefix}_rcp{rcp}_{year}_sobel.txt".format(prefix=prefix, rcp=rcp, year=year)


def read_sobel_file(filename):
    data = np.loadtxt(filename, usecols=(1))
    return data

categories = ["Climate", "Surface", "Ocean", "Ice Dynamics"]
category_col_dict = {"Climate": "#81c77f",
                     "Surface": "#886c62",
                     "Ocean": "#beaed4",
                     "Ice Dynamics": "#dcd588"}

parser = ArgumentParser()
parser.description = "Generate tables for the paper"
parser.add_argument("FILE", nargs="*")
options = parser.parse_args()
ifiles = options.FILE
prefix = "les_gcm"

years = range(2015, 2500)
nt = len(years)
nc = len(categories)
climate = np.zeros(nt)
surface = np.zeros(nt)
ocean = np.zeros(nt)
ice = np.zeros(nt)

for rcp in ["26", "45"]:
    for t, year in enumerate(years):
        filename = input_filename(prefix, rcp, year)
        mdata = read_sobel_file(filename)
        climate[t] = mdata[0] + mdata[3]
        surface[t] = mdata[1] + mdata[2] + mdata[4]
        ocean[t] = mdata[5] + mdata[6] + mdata[7] + mdata[8]
        ice[t] = mdata[9] + mdata[10]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.bar(years, climate, color=category_col_dict[categories[0]], label=categories[0])
    ax.bar(years, surface, bottom=climate, color=category_col_dict[categories[1]], label=categories[1])
    ax.bar(years, ocean, bottom=climate + surface, color=category_col_dict[categories[2]], label=categories[2])
    ax.bar(years, ice, bottom=climate + surface + ocean, color=category_col_dict[categories[3]], label=categories[3])
    plt.legend()
    
