#!/usr/bin/env python

"""This script generates rows for two LaTeX tables: sea level
contributions in cm for different runs and RCP scenarios and mass rate
changes for the control run and different RCP scenarios.
"""

from cdo import Cdo
from cf_units import date2num, Unit
import netCDF4
import numpy as np
from datetime import datetime
import os.path

from argparse import ArgumentParser

def input_filename(prefix, run, resolution, rcp):
    "Return a file name for the given run, resolution, and RCP scenation."
    filename = "ts_gris_g{resolution}m_v3a_rcp_{rcp}_id_{run}_0_1000.nc".format(resolution=resolution,
                                                                                run=run,
                                                                                rcp=rcp)

    return os.path.join(prefix, filename)

def extract_time_mean(filename, year, window_width):
    "Extract the time mean of the data in filename, using the window window_width years wide centered on a given year."
    delta = window_width / 2
    command = "-selyear,{min_year}/{max_year} {filename}".format(min_year=year-delta,
                                                                 max_year=year+delta,
                                                                 filename=filename)
    return Cdo().timmean(input=command, returnCdf=True)

def sl_cm(mass, units):
    "Return sea level contribution in cm, assuming one mm contribution per gigaton."
    return (Unit(units).convert(mass, "Gt") / 365.0) / 10.0

def sl_contribution_cm(filename, years, variable="limnsw"):
    """Compute sea level contributions for given years. The first year is
    used to normalize the output."""

    f = netCDF4.Dataset(filename)

    t = f.variables["time"]
    v = f.variables[variable]

    try:
        requested_times = date2num([datetime(y, 1, 1) for y in years], t.units, "365_day")
    except:
        # the ensstat case: times are in 365-day years since 2008
        requested_times = np.array(years) * 365 * 86400

    data = np.interp(requested_times, t[:], v[:])

    return sl_cm(data[0] - data[1:], v.units)

def latex_table_row(label, array):
    "Format a table row for a latex table."
    return label + " & " + " & ".join(["{:.0f}".format(x) for x in array]) + " \\\\"

def ensstat_filename(prefix, percentile, rcp):
    "Return the ensemble statistics file name for given percentile and RCP scenario."
    filename = "enspctl{percentile}_gris_g1800m_v3a_rcp_{rcp}_0_1000.nc".format(rcp=rcp, percentile=percentile)

    return os.path.join(prefix, filename)

def les_table_row(prefix, label="LES"):
    "Gather large ensemble statistics into a row for the sea level contribution table."

    # years since 2008
    years = np.array([8, 100, 200, 500, 1000]) - 8
    rcps = [26, 45, 85]

    result = []
    for percentile in [84, 16]:
        data = []
        for rcp in rcps:
            filename = ensstat_filename(prefix, percentile, rcp)

            data.append(sl_contribution_cm(filename, years))

        result.append(np.hstack(data))

    table_row = label
    for x in np.array(result).T:
        table_row += " & " + "{:.0f}--{:.0f}".format(x[0], x[1])
    table_row += " \\\\"

    return table_row

def sea_level_contribution_table(prefix, ensstat_prefix):
    years = [2008, 2100, 2200, 2500, 3000]
    rcps = [26, 45, 85]
    rows = [
        {"label"      : "CTRL" ,
         "resolution" : 900,
         "run"        : "CTRL"},
        {"label"      : "NTRL" ,
         "resolution" : 900,
         "run"        : "NTRL"},
        {"label"      : "G1800",
         "resolution" : 1800,
         "run"        : "CTRL"},
        {"label"      : "G3600",
         "resolution" : 3600,
         "run"        : "CTRL"},
        {"label"      : "G4500",
         "resolution" : 4500,
         "run"        : "CTRL"},
        {"label"      : "G9000",
         "resolution" : 9000,
         "run"        : "CTRL"},
        {"label"      : "G18000",
         "resolution" : 18000,
         "run"        : "CTRL"},
    ]

    result = [les_table_row(ensstat_prefix)]

    for row in rows:
        data = []
        for rcp in rcps:
            filename = input_filename(prefix, row["run"], row["resolution"], rcp)

            data.append(sl_contribution_cm(filename, years))

        result.append(latex_table_row(row["label"], np.hstack(data)))

    return result

def convert_units(variable):
    "Convert units to Gt/year."
    return Unit(variable.units).convert(variable[0], "Gt year-1")

def mass_rate_table(prefix, window_width=20, years=[2100, 2200, 2500, 3000]):
    rows = [
        {"label"    : "total",
         "variable" : "tendency_of_ice_mass",
         "sign"     : 1.0},
        {"label"    : "snowfall",
         "variable" : "surface_accumulation_rate",
         "sign"     : 1.0},    # positive means mass loss
        {"label"    : "runoff",
         "variable" : "surface_runoff_rate",
         "sign"     : -1.0},    # positive means mass loss
        {"label"    : "discharge",
         "variable" : "tendency_of_ice_mass_due_to_discharge",
         "sign"     : 1.0},
        {"label"    : "basal mass balance",
         "variable" : "tendency_of_ice_mass_due_to_basal_mass_flux",
         "sign"     : 1.0}
    ]

    rcps = [26, 45, 85]

    # create the header
    result = [latex_table_row("", years * len(rcps))]

    for row in rows:
        variable = row["variable"]
        label    = row["label"]
        sign     = row["sign"]

        data = []

        for rcp in rcps:
            filename = input_filename(prefix, "CTRL", 900, rcp)

            for year in years:
                f = extract_time_mean(filename, year, window_width)

                data.append(sign * convert_units(f.variables[variable]))

        result.append(latex_table_row(label, data))

    return result

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.description = "Generate tables for the paper"
    parser.add_argument("--prefix", dest="prefix",
                        help="the directory containing output files from the study",
                        default="/import/c1/ICESHEET/aaschwanden/pism-gris/stability/2017_12_ctrl/scalar/")
    parser.add_argument("--ensstat_prefix", dest="ensstat_prefix",
                        help="the directory containing ensemble stats from the study",
                        default="/import/c1/ICESHEET/aaschwanden/pism-gris/stability/2018_01_les/scalar_ensstat/")
    options = parser.parse_args()

    print("% sea level contribution table")
    for row in sea_level_contribution_table(options.prefix, options.ensstat_prefix):
        print(row)

    print("% mass rate of change table, in Gt/year, for RCP 2.6, 4.5, 8.5 as in the previous table")
    for row in mass_rate_table(options.prefix):
        print(row)
