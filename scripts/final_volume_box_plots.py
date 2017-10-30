#!/usr/bin/env python

import netCDF4 as NC
import pylab as plt
from argparse import ArgumentParser
import numpy as np
import re

parser = ArgumentParser()
parser.add_argument("FILE", nargs='*')
parser.add_argument("--parameters", type=str, default="../latin_hypercube/lhs_samples_20171026.csv")

options = parser.parse_args()

def read_volumes(files):
    "Read final ice volumes, run IDs, and RCPs from provided files."

    data = np.array(np.zeros(len(files)), dtype=[("ID", "i4"),
                                                 ("RCP", "i4"),
                                                 ("volume", "f8")])

    for j, filename in enumerate(options.FILE):
        regexp = re.compile(r".+rcp_([0-9]+)_id_([0-9]+)")

        match = re.match(regexp, filename)

        data[j]["RCP"] = int(match.group(1))
        data[j]["ID"] = int(match.group(2))

        F = NC.Dataset(filename, "r")
        data[j]["volume"] = F.variables["limnsw"][-1]
        F.close()

    return data


def read_parameters(filename):
    "Read parameters from a csv file "

    parameters = np.loadtxt(options.parameters, delimiter=",", skiprows=1)

    fields = ["FICE","FSNOW","PRS","RFR","OCM","OCS","TCT","VCM","PPQ","SIAE"]
    dtype = np.dtype(zip(fields, ["f8"] * len(fields)))

    d = {}
    for p in parameters:
        d[int(p[0])] = tuple(p[1:])

    return d, dtype

def combine(volumes, parameters):
    "Combine ice volumes from runs with parameters that produced them."

    P, P_dtype = parameters

    dtype = volumes.dtype.descr + P_dtype.descr

    result = np.array(np.zeros(len(volumes)),
                      dtype=dtype)

    for j, v in enumerate(volumes):
        result[j] = tuple(v) + P[v["ID"]]

    return result

def plot_runs(data, run_mask, title):

    if run_mask is None:
        run_mask = np.zeros_like(data, dtype=bool)

    N_runs = len(data)

    fig = plt.figure(figsize=(8,8))
    plt.clf()

    fig.suptitle(title)

    xs = np.arange(N_runs)
    xlabels = data["ID"]

    plots = [x[0] for x in data[0].dtype.descr[2:]]
    # plots = ["volume", "FSNOW", "FICE"]
    N_plots = len(plots)

    for j,field in enumerate(plots):
        plt.subplot(N_plots, 1, j+1)

        all_data = data[field]
        masked_data = data[run_mask][field]

        plt.boxplot([masked_data, all_data], vert=False, labels=("selected", "all"))

        # plt.scatter(xs, np.ma.array(data[field], mask=run_mask), s=1)
        # plt.scatter(xs, np.ma.array(data[field], mask=1-run_mask), s=1)
        plt.ylabel(field)

def analyze(data, rcp):
    """Produce plots to analyze final ice volumes vs. parameter choices
    for a given RCP scenario."""

    V_median = np.median(data["volume"])

    V_std = np.abs(np.median(data["volume"]) - np.percentile(data["volume"], 34))

    plot_runs(data, data["volume"] > V_median + V_std,
              "RCP {}, V > V_median + sigma".format(rcp))

    plt.savefig("rcp_{}_high.png".format(rcp))

    plot_runs(data, data["volume"] < V_median - V_std,
              "RCP {}, V < V_median - sigma".format(rcp))

    plt.savefig("rcp_{}_low.png".format(rcp))

    plt.figure()
    plt.scatter(data["FICE"], data["volume"], s=1)
    plt.xlabel("FICE")
    plt.ylabel("volume")
    plt.title("RCP {}".format(rcp))
    plt.savefig("rcp_{}_fice_vs_volume.png".format(rcp))

if __name__ == "__main__":

    volumes = read_volumes(options.FILE)

    parameters = read_parameters(options.parameters)

    data = combine(volumes, parameters)

    # for rcp in (26, 45, 85):
    for rcp in [26]:
        analyze(data[data["RCP"] == rcp], rcp)

    plt.show()
