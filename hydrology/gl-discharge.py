import numpy as np
from netCDF4 import Dataset as NC
import pylab as plt
import pandas as pd
from cftime import num2date


def set_size(w, h, ax=None):
    """ w, h: width, height in inches """

    if not ax:
        ax = plt.gca()
    l = ax.figure.subplotpars.left
    r = ax.figure.subplotpars.right
    t = ax.figure.subplotpars.top
    b = ax.figure.subplotpars.bottom
    figw = float(w)/(r-l)
    figh = float(h)/(t-b)
    ax.figure.set_size_inches(figw, figh)


basins = ('CW', 'NE', 'NO', 'NW', 'SE', 'SW')

normalize = False

for basin in basins:
    nc = NC('b_{basin}_ex_g3600m_water_routing_DMI-HIRHAM5_GL2_ERAI_1980_2014_dm.nc'.format(basin=basin), 'r')
    time = nc.variables['time']
    time_units = time.units
    time_calendar = time.calendar

    dates = num2date(time[:], time_units, calendar=time_calendar)
    input_flux = np.squeeze(nc.variables['tendency_of_subglacial_water_mass_due_to_input'][:])
    gl_flux = np.squeeze(nc.variables['tendency_of_subglacial_water_mass_at_grounding_line'][:])
    gl_flux_ts = pd.Series(data=gl_flux, index=time, name='gl_flux')
    input_flux_ts = pd.Series(data=gl_flux, index=time, name='input_flux')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    if normalize:
        ax.plot_date(dates, input_flux / np.max(input_flux), lw=0.4, ls='solid', markersize=0, label='Input Flux')
        ax.plot_date(dates, -gl_flux / np.max(-gl_flux), lw=0.5, ls='solid', markersize=0, label='Grounding Line Flux')
    else:
        ax.plot_date(dates, input_flux, lw=0.4, ls='solid', markersize=0, label='Input Flux')
        ax.plot_date(dates, -gl_flux, lw=0.5, ls='solid', markersize=0, label='Grounding Line Flux')

    plt.title('basin {basin}'.format(basin=basin))
    legend = plt.legend()
    legend.get_frame().set_linewidth(0.0)
    legend.get_frame().set_alpha(0.0)

    set_size(6, 3)

    out_file = 'gl_flux_{basin}.pdf'.format(basin=basin)
    print("  - writing image %s ..." % out_file)
    fig.savefig(out_file, bbox_inches='tight')
