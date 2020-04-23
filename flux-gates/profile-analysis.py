#!/usr/bin/env python
# Copyright (C) 2014-2019 Andy Aschwanden

import ogr
import osr
import os
from unidecode import unidecode
import itertools
import codecs
import operator
import numpy as np
import pylab as plt
import matplotlib as mpl
from colorsys import rgb_to_hls, hls_to_rgb
import matplotlib.cm as cmx
import matplotlib.colors as mplcolors
from argparse import ArgumentParser
import pandas as pa
from palettable import colorbrewer
import statsmodels.api as sm
from netCDF4 import Dataset as NC
import re

try:
    import pypismtools.pypismtools as ppt
except:
    import pypismtools as ppt

import cf_units


def reverse_enumerate(iterable):
    """
    Enumerate over an iterable in reverse order while retaining proper indexes
    """

    return zip(reversed(range(len(iterable))), reversed(iterable))


def get_rmsd(a, b, w=None):
    """
    Returns the (weighted) root mean square differences between a and b.

    Parameters
    ----------
    a, b : array_like
    w : weights

    Returns
    -------
    rmsd : scalar
    """

    if w is None:
        w = np.ones_like(a)
    # If observations or errors have missing values, don't count them
    c = (a.ravel() - b.ravel()) / w.ravel()
    if isinstance(c, np.ma.MaskedArray):
        # count all non-masked elements
        N = c.count()
        # reduce to non-masked elements
        return np.sqrt(np.linalg.norm(np.ma.compressed(c), 2) ** 2.0 / N), N
    else:
        N = c.shape[0]
        return np.sqrt(np.linalg.norm(c, 2) ** 2.0 / N), N


class FluxGate(object):

    """
    A class for FluxGates.

    Parameters
    ----------
    pos_id: int, pos of gate in array
    gate_name: string, name of flux gate
    gate_id: int, gate identification
    profile_axis: 1-d array, profile axis values
    profile_axis_units: string, udunits unit of axis
    profile_axis_name: string, descriptive name of axis

    """

    def __init__(
        self,
        pos_id,
        gate_name,
        gate_id,
        profile_axis,
        profile_axis_units,
        profile_axis_name,
        clon,
        clat,
        flightline,
        glaciertype,
        flowtype,
        *args,
        **kwargs
    ):
        super(FluxGate, self).__init__(*args, **kwargs)
        self.pos_id = pos_id
        self.gate_name = gate_name
        self.gate_id = gate_id
        self.profile_axis = profile_axis
        self.profile_axis_units = profile_axis_units
        self.profile_axis_name = profile_axis_name
        self.clon = clon
        self.clat = clat
        self.flightline = flightline
        self.glaciertype = glaciertype
        self.flowtype = flowtype
        self.best_rmsd_exp_id = None
        self.best_rmsd = None
        self.best_corr_exp_id = None
        self.best_corr = None
        self.corr = None
        self.corr_units = None
        self.experiments = []
        self.exp_counter = 0
        self.has_observations = None
        self.has_fluxes = None
        self.has_stats = None
        self.linear_trend = None
        self.linear_bias = None
        self.linear_r2 = None
        self.linear_p = None
        self.N_corr = None
        self.N_rmsd = None
        self.observed_flux = None
        self.observed_flux_units = None
        self.observed_flux_error = None
        self.observed_mean = None
        self.observed_mean_units = None
        self.p_ols = None
        self.r2 = None
        self.r2_units = None
        self.rmsd = None
        self.rmsd_units = None
        self.S = None
        self.S_units = None
        self.sigma_obs = None
        self.sigma_obs_N = None
        self.sigma_obs_units = None
        self.varname = None
        self.varname_units = None

    def __repr__(self):
        return "FluxGate"

    def add_experiment(self, data):
        """
        Add an experiment to FluxGate

        """

        print(("      adding experiment to flux gate {0}".format(self.gate_name)))
        pos_id = self.pos_id
        fg_exp = FluxGateExperiment(data, pos_id)
        self.experiments.append(fg_exp)
        if self.varname is None:
            self.varname = data.varname
        if self.varname_units is None:
            self.varname_units = data.varname_units
        self.exp_counter += 1

    def add_observations(self, data):
        """
        Add observations to FluxGate

        """

        print(("      adding observations to flux gate {0}".format(self.gate_name)))
        pos_id = self.pos_id
        fg_obs = FluxGateObservations(data, pos_id)
        self.observations = fg_obs
        if self.has_observations is not None:
            print(("Flux gate {0} already has observations, overriding".format(self.gate_name.encode("utf-8"))))
        self.has_observations = True

    def calculate_fluxes(self):
        """
        Calculate fluxes

        """

        if self.has_observations:
            self._calculate_observed_flux()
        self._calculate_experiment_fluxes()
        self.has_fluxes = True

    def calculate_stats(self):
        """
        Calculate statistics
        """

        if not self.has_fluxes:
            self.calculate_fluxes()
        corr = {}
        N_corr = {}
        N_rmsd = {}
        p_ols = {}
        r2 = {}
        rmsd = {}
        S = {}
        for exp in self.experiments:
            id = exp.id
            x = np.squeeze(self.profile_axis)
            obs_vals = np.squeeze(self.observations.values)
            sigma_obs = self.sigma_obs
            # mask values where obs is zero
            obs_vals = np.ma.masked_where(obs_vals == 0, obs_vals)
            if isinstance(obs_vals, np.ma.MaskedArray):
                obs_vals = obs_vals.filled(0)
            exp_vals = np.squeeze(self.experiments[id].values)
            # Calculate root mean square difference (RMSD), convert units
            my_rmsd, my_N_rmsd = get_rmsd(exp_vals, obs_vals)
            i_units = self.varname_units
            o_units = v_o_units
            i_units_cf = cf_units.Unit(i_units)
            o_units_cf = cf_units.Unit(o_units)
            rmsd[id] = i_units_cf.convert(my_rmsd, o_units_cf)
            N_rmsd[id] = my_N_rmsd
            obsS = pa.Series(data=obs_vals, index=x)
            expS = pa.Series(data=exp_vals, index=x)
            p_ols[id] = sm.OLS(expS, sm.add_constant(obsS), missing="drop").fit()
            r2[id] = p_ols[id].rsquared
            corr[id] = obsS.corr(expS)
        best_rmsd_exp_id = sorted(p_ols, key=lambda x: rmsd[x], reverse=False)[0]
        best_corr_exp_id = sorted(p_ols, key=lambda x: corr[x], reverse=True)[0]
        self.p_ols = p_ols
        self.best_rmsd_exp_id = best_rmsd_exp_id
        self.best_rmsd = rmsd[best_rmsd_exp_id]
        self.best_corr_exp_id = best_corr_exp_id
        self.best_corr = corr[best_corr_exp_id]
        self.rmsd = rmsd
        self.rmsd_units = v_o_units
        self.N_rmsd = N_rmsd
        self.S = S
        self.S_units = "1"
        self.r2 = r2
        self.r2_units = "1"
        self.corr = corr
        self.corr_units = "1"
        self.has_stats = True
        self.observed_mean = np.mean(self.observations.values)
        self.observed_mean_units = self.varname_units

    def _calculate_observed_flux(self):
        """
        Calculate observed flux

        Calculate observed flux using trapezoidal rule. If observations have
        asscociated errors, the error in observed flux is calculated as well.
        """

        x = self.profile_axis
        y = self.observations.values
        x_units = self.profile_axis_units
        y_units = self.varname_units
        int_val = self._line_integral(y, x)
        # Here we need to directly access udunits2 since we want to
        # multiply units
        if vol_to_mass:
            i_units_cf = cf_units.Unit(x_units) * cf_units.Unit(y_units) * cf_units.Unit(ice_density_units)
        else:
            i_units_cf = cf_units.Unit(x_units) * cf_units.Unit(y_units)
        o_units_cf = cf_units.Unit(v_flux_o_units)
        o_units_str = v_flux_o_units_str
        o_val = i_units_cf.convert(int_val, o_units_cf)
        observed_flux = o_val
        observed_flux_units = o_units_str
        if self.observations.has_error:
            y = self.observations.error
            int_val = self._line_integral(y, x)
            i_error = int_val
            o_error = i_units_cf.convert(i_error, o_units_cf)
            error_norm, N = get_rmsd(y, np.zeros_like(y, dtype="float32"))
            self.sigma_obs = error_norm
            self.sigma_obs_N = N
            self.sigma_obs_units = self.varname_units
            self.observed_flux_error = o_error

        self.observed_flux = observed_flux
        self.observed_flux_units = observed_flux_units

    def _calculate_experiment_fluxes(self):
        """
        Calculate experiment fluxes

        Calculated experiment fluxes using trapeziodal rule.

        """

        experiment_fluxes = {}
        experiment_fluxes_units = {}
        for exp in self.experiments:
            id = exp.id
            x = self.profile_axis
            y = exp.values
            x_units = self.profile_axis_units
            y_units = self.varname_units
            int_val = self._line_integral(y, x)
            # Here we need to directly access udunits2 since we want to
            # multiply units
            if vol_to_mass:
                i_units = cf_units.Unit(x_units) * cf_units.Unit(y_units) * cf_units.Unit(ice_density_units)
            else:
                i_units = cf_units.Unit(x_units) * cf_units.Unit(y_units)
            o_units = cf_units.Unit(v_flux_o_units)
            o_units_str = v_flux_o_units_str
            o_val = i_units.convert(int_val, o_units)
            experiment_fluxes[id] = o_val
            experiment_fluxes_units[id] = o_units_str
            config = exp.config
            my_exp_str = ", ".join(
                [
                    "=".join([params_dict[key]["abbr"], params_dict[key]["format"].format(config.get(key))])
                    for key in label_params
                ]
            )
        self.experiment_fluxes = experiment_fluxes
        self.experiment_fluxes_units = experiment_fluxes_units

    def length(self):
        """
        Return length of the profile, rounded to the nearest meter.
        """

        return np.around(self.profile_axis.max())

    def return_gate_flux_str(self, islast):
        """
        Returns a LaTeX-table compatible string
        """

        if not self.has_stats:
            self.calculate_stats()
        gate_name = self.gate_name
        p_ols = self.p_ols
        observed_flux = self.observed_flux
        observed_flux_units = self.observed_flux_units
        experiment_fluxes = self.experiment_fluxes
        experiment_fluxes_units = self.experiment_fluxes_units
        observed_flux_error = self.observed_flux_error
        if observed_flux_error:
            obs_str = "$\pm$".join(["{:2.1f}".format(observed_flux), "{:2.1f}".format(observed_flux_error)])
        else:
            obs_str = "{:2.1f}".format(observed_flux)
        exp_str = " & ".join(
            [
                "".join(["{:2.1f}".format(experiment_fluxes[key]), "({:1.2f})".format(p_ols[key].rsquared)])
                for key in experiment_fluxes
            ]
        )
        if islast:
            gate_str = " & ".join([gate_name, obs_str, "", exp_str])
        else:
            gate_str = " & ".join([gate_name, obs_str, "", " ".join([exp_str, r" \\"])])
        return gate_str

    def return_gate_flux_str_short(self):
        """
        Returns a LaTeX-table compatible string
        """

        if not self.has_stats:
            self.calculate_stats()
        gate_name = self.gate_name
        p_ols = self.p_ols
        observed_flux = self.observed_flux
        observed_flux_units = self.observed_flux_units
        experiment_fluxes = self.experiment_fluxes
        experiment_fluxes_units = self.experiment_fluxes_units
        observed_flux_error = self.observed_flux_error
        if observed_flux_error:
            obs_str = "$\pm$".join(["{:2.1f}".format(observed_flux), "{:2.1f}".format(observed_flux_error)])
        else:
            obs_str = "{:2.1f}".format(observed_flux)
        gate_str = " & ".join([gate_name, obs_str])
        return gate_str

    def _line_integral(self, y, x):
        """
        Return line integral using the composite trapezoidal rule

        Parameters
        ----------
        y: 1-d array_like
           Input array to integrate
        x: 1-d array_like
           Spacing between elements

        Returns
        -------
        trapz : float
                Definite integral as approximated by trapezoidal rule.
        """

        # Due to the variable length of profiles, we have masked arrays, with
        # masked values at the profile end. We can assume zero error here,
        # since it's not used for the computation

        if isinstance(y, np.ma.MaskedArray):
            y = y.filled(0)

        return float(np.squeeze(np.trapz(y, x)))

    def make_line_plot(self, **kwargs):
        """
        Make a plot.

        Make a line plot along a flux gate.
        """

        gate_name = self.gate_name
        experiments = self.experiments
        profile_axis = self.profile_axis
        profile_axis_name = self.profile_axis_name
        profile_axis_units = self.profile_axis_units
        i_units_cf = cf_units.Unit(profile_axis_units)
        o_units_cf = cf_units.Unit(profile_axis_out_units)
        profile_axis_out = i_units_cf.convert(profile_axis, o_units_cf)
        varname = self.varname
        v_units = self.varname_units
        has_observations = self.has_observations
        if not self.has_fluxes:
            self.calculate_fluxes()
        if has_observations:
            self.calculate_stats()

        labels = []
        fig = plt.figure()
        ax = fig.add_subplot(111)
        if has_observations:
            obs = self.observations
            config = obs.config
            if legend == "dem":
                if obs.has_error:
                    label = "obs: {:6.1f}$\pm${:4.1f}".format(self.observed_flux, self.observed_flux_error)
                else:
                    label = config["dem"]
            else:
                if obs.has_error:
                    label = "{:6.1f}$\pm${:4.1f}".format(self.observed_flux, self.observed_flux_error)
                else:
                    label = "{:6.1f}".format(self.observed_flux)
            has_error = obs.has_error
            i_vals = obs.values
            i_units_cf = cf_units.Unit(v_units)
            o_units_cf = cf_units.Unit(v_o_units)
            obs_o_vals = i_units_cf.convert(i_vals, o_units_cf)
            obs_max = np.max(obs_o_vals)
            if has_error:
                i_vals = obs.error
                obs_error_o_vals = i_units_cf.convert(i_vals, o_units_cf)
                ax.fill_between(
                    profile_axis_out, obs_o_vals - obs_error_o_vals, obs_o_vals + obs_error_o_vals, color="0.85"
                )

            if simple_plot:
                ax.plot(profile_axis_out, obs_o_vals, "-", color="0.35", label=label)
            else:
                ax.plot(profile_axis_out, obs_o_vals, "-", color="0.5")
                ax.plot(
                    profile_axis_out,
                    obs_o_vals,
                    dash_style,
                    color=obscolor,
                    markeredgewidth=markeredgewidth,
                    markeredgecolor=markeredgecolor,
                    label=label,
                )

        ne = len(experiments)
        lines_d = []
        lines_c = []
        # We need to carefully reverse list and properly order
        # handles and labels to have first experiment plotted on top
        for k, exp in enumerate(reversed(experiments)):
            i_vals = exp.values
            i_units_cf = cf_units.Unit(v_units)
            o_units_cf = cf_units.Unit(v_o_units)
            exp_o_vals = i_units_cf.convert(i_vals, o_units_cf)
            if normalize:
                exp_max = np.max(exp_o_vals)
                exp_o_vals *= obs_max / exp_max
            if "label_param_list" in list(kwargs.keys()):
                params = kwargs["label_param_list"]
                config = exp.config
                id = exp.id
                exp_str = ", ".join(
                    [
                        "=".join([params_dict[key]["abbr"], params_dict[key]["format"].format(config.get(key))])
                        for key in params
                    ]
                )
                if has_observations:
                    if legend == "dem":
                        label = config.get("dem")
                else:
                    label = config.get("dem")
                labels.append(label)

            my_color = my_colors[k]
            my_color_light = my_colors_light[k]
            if simple_plot:
                line_c, = ax.plot(profile_axis_out, exp_o_vals, color=my_color, label=label)
                line_d, = ax.plot(profile_axis_out, exp_o_vals, color=my_color)
            else:
                line_c, = ax.plot(profile_axis_out, exp_o_vals, "-", color=my_color, alpha=0.5)
                line_d, = ax.plot(
                    profile_axis_out,
                    exp_o_vals,
                    dash_style,
                    color=my_color,
                    markeredgewidth=markeredgewidth,
                    markeredgecolor=markeredgecolor,
                    label=label,
                )
            labels.append(label)
            lines_d.append(line_d)
            lines_c.append(line_c)

        if (x_lim_min is not None) or( x_lim_max is not None):
            ax.set_xlim(x_lim_min, int(x_lim_max))
        else:
            ax.set_xlim(0, np.max(profile_axis_out))
        xlabel = "{0} ({1})".format(profile_axis_name, profile_axis_out_units)
        ax.set_xlabel(xlabel)
        if varname in list(var_name_dict.keys()):
            v_name = var_name_dict[varname]
        else:
            v_name = varname
        ylabel = "{0} ({1})".format(v_name, v_o_units_str)
        ax.set_ylabel(ylabel)
        handles, labels = ax.get_legend_handles_labels()
        ordered_handles = handles[:0:-1]
        ordered_labels = labels[:0:-1]
        ordered_handles.insert(0, handles[0])
        ordered_labels.insert(0, labels[0])
        if legend != "none":
            if (legend == "dem") :
                lg = ax.legend(
                    ordered_handles,
                    ordered_labels,
                    loc="upper right",
                    shadow=True,
                    numpoints=numpoints,
                    bbox_to_anchor=(0, 0, 1, 1),
                    bbox_transform=plt.gcf().transFigure,
                )
            else:
                lg = ax.legend(
                    ordered_handles,
                    ordered_labels,
                    loc="upper right",
                    title="{} ({})".format(flux_type, self.experiment_fluxes_units[0]),
                    shadow=True,
                    numpoints=numpoints,
                    bbox_to_anchor=(0, 0, 1, 1),
                    bbox_transform=plt.gcf().transFigure,
                )
            fr = lg.get_frame()
            fr.set_lw(legend_frame_width)
        # Replot observations
        if has_observations:
            if simple_plot:
                ax.plot(profile_axis_out, obs_o_vals, "-", color="0.35")
            else:
                ax.plot(profile_axis_out, obs_o_vals, "-", color="0.5")
                ax.plot(profile_axis_out, obs_o_vals, dash_style, color=obscolor, markeredgewidth=markeredgewidth, markeredgecolor=markeredgecolor,
)
        if (y_lim_min is not None) or( y_lim_max is not None):
            ax.set_ylim(bottom=y_lim_min, top=y_lim_max)
        if plot_title:
            plt.title(gate_name, loc="left")

        if normalize:
            gate_name = "_".join([unidecode(gate.gate_name), varname, "normalized", "profile"])
        else:
            gate_name = "_".join([unidecode(gate.gate_name), varname, "profile"])
        outname = os.path.join(odir, ".".join([gate_name, "pdf"]).replace(" ", "_"))
        print(("Saving {0}".format(outname)))
        fig.tight_layout()
        fig.savefig(outname)
        plt.close(fig)


class FluxGateExperiment(object):
    def __init__(self, data, pos_id, *args, **kwargs):
        super(FluxGateExperiment, self).__init__(*args, **kwargs)
        self.values = data.values[pos_id, Ellipsis]
        self.config = data.config
        self.id = data.id

    def __repr__(self):
        return "FluxGateExperiment"


class FluxGateObservations(object):
    def __init__(self, data, pos_id, *args, **kwargs):
        super(FluxGateObservations, self).__init__(*args, **kwargs)
        self.values = data.values[pos_id, Ellipsis]
        self.config = data.config
        self.has_error = None
        if data.has_error:
            self.error = data.error[pos_id, Ellipsis]
            self.has_error = True

    def __repr__(self):
        return "FluxGateObservations"


class Dataset(object):

    """
    A base class for Experiments or Observations.

    Constructor opens netCDF file, attaches pointer to nc instance.

    """

    def __init__(self, filename, varname, *args, **kwargs):
        super(Dataset, self).__init__(*args, **kwargs)
        print(("  opening NetCDF file %s ..." % filename))
        try:
            nc = NC(filename, "r")
        except:
            print(("ERROR:  file '%s' not found or not NetCDF format ... ending ..." % filename))
            import sys

            sys.exit(1)

        for name in nc.variables:
            v = nc.variables[name]
            if getattr(v, "standard_name", "") == varname:
                print(("variabe {0} found by its standard_name {1}".format(name, varname)))
                varname = name

        self.values = nc.variables[varname][:]
        self.varname_units = nc.variables[varname].units
        self.varname = varname
        self.nc = nc

    def __repr__(self):
        return "Dataset"

    def __del__(self):
        # Close open file
        self.nc.close()


class ExperimentDataset(Dataset):

    """
    A derived class for experiments

    A derived class for handling PISM experiments.

    Experiments are identified by id. Config and run_stats
    attributes are attached to object as config dictionary.

    """

    def __init__(self, id, *args, **kwargs):
        super(ExperimentDataset, self).__init__(*args, **kwargs)

        print("Experiment {}".format(id))
        self.id = id
        self.config = dict()
        for v in ["config"]:
            if v in self.nc.variables:
                ncv = self.nc.variables[v]
                for attr in ncv.ncattrs():
                    self.config[attr] = getattr(ncv, attr)
            else:
                print("Variable {} not found".format(v))
                
    def __repr__(self):
        return "ExperimentDataset"


class ObservationsDataset(Dataset):

    """
    A derived class for observations.

    A derived class for handling observations.

    """

    def __init__(self, *args, **kwargs):
        super(ObservationsDataset, self).__init__(*args, **kwargs)
        self.has_error = None
        try:
            self.clon = self.nc.variables["clon"][:]
        except:
            self.clon = None
        try:
            self.clat = self.nc.variables["clat"][:]
        except:
            self.clat = None
        try:
            self.flightline = self.nc.variables["flightline"][:]
        except:
            self.flightline = None
        try:
            self.glaciertype = self.nc.variables["glaciertype"][:]
        except:
            self.glaciertype = None
        try:
            self.flowtype = self.nc.variables["flowtype"][:]
        except:
            self.flowtype = None
        varname = self.varname
        error_varname = "_".join([varname, "error"])
        if error_varname in list(self.nc.variables.keys()):
            self.error = self.nc.variables[error_varname][:]
            self.has_error = True
        self.config = dict()
        for v in ["config"]:
            if v in self.nc.variables:
                ncv = self.nc.variables[v]
                for attr in ncv.ncattrs():
                    self.config[attr] = getattr(ncv, attr)
            else:
                print("Variable {} not found".format(v))

    def __repr__(self):
        return "ObservationsDataset"







# ##############################################################################
# MAIN
# ##############################################################################

if __name__ == "__main__":

    __spec__ = None


    # Set up the option parser
    parser = ArgumentParser()
    parser.description = "Analyze flux gates. Used for 'Complex Greenland Outlet Glacier Flow Captured'."
    parser.add_argument("FILE", nargs="*")
    parser.add_argument("--aspect_ratio", dest="aspect_ratio", type=float, help='''Plot aspect ratio"''', default=0.8)
    parser.add_argument(
        "--colormap",
        dest="colormap",
        nargs=1,
        help="""Name of matplotlib colormap""",
        default="tab20c"
    )
    parser.add_argument(
        "--label_params",
        dest="label_params",
        help='''comma-separated list of parameters that appear in the legend,
                      e.g. "sia_enhancement_factor"''',
        default="basal_resistance.pseudo_plastic.q,basal_yield_stress.mohr_coulomb.till_effective_fraction_overburden,stress_balance.sia.enhancement_factor,basal_yield_stress.mohr_coulomb.topg_to_phi.phi_min,flow_law.gpbld.water_frac_observed_limit,basal_yield_stress.mohr_coulomb.topg_to_phi.topg_min,basal_yield_stress.mohr_coulomb.topg_to_phi.topg_max",
    )
    parser.add_argument(
        "--normalize",
        dest="normalize",
        action="store_true",
        help="Normalize experiments by muliplying with max(obs)/max(experiment)",
        default=False,
    )
    parser.add_argument(
        "--obs_file", dest="obs_file", help="""Profile file with observations. Default is None""", default=None
    )
    parser.add_argument(
        "--export_table_file",
        dest="table_file",
        help="""If given, fluxes are exported to latex table. Default is None""",
        default=None,
    )
    parser.add_argument(
        "--no_figures", dest="make_figures", action="store_false", help="Do not make profile figures", default=True
    )
    parser.add_argument(
        "--do_regress", dest="do_regress", action="store_true", help="Make grid resolution regression plots", default=False
    )
    parser.add_argument(
        "--legend",
        dest="legend",
        choices=["dem"],
        help="Controls the legend",
        default="dem",
    )
    parser.add_argument("--o_dir", dest="odir", help="output directory. Default: current directory", default="figures")
    parser.add_argument(
        "--plot_title", dest="plot_title", action="store_true", help="Plots the flux gate name as title", default=False
    )
    parser.add_argument(
        "--simple_plot", dest="simple_plot", action="store_true", help="Make simple line plot", default=False
    )
    parser.add_argument("--no_legend", dest="plot_legend", action="store_false", help="Don't plot a legend", default=True)
    parser.add_argument(
        "-p",
        "--print_size",
        dest="print_mode",
        choices=["onecol", "medium", "twocol", "height", "presentation", "small_font", "large_font", "50mm", "72mm"],
        help="sets figure size and font size, available options are: \
                        'onecol','medium','twocol','presentation'",
        default="medium",
    )
    parser.add_argument(
        "-r",
        "--output_resolution",
        dest="out_res",
        help="""
                      Graphics resolution in dots per inch (DPI), default
                      = 300""",
        default=300,
    )
    parser.add_argument("--x_lim", dest="x_lim", nargs=2, help="""X lims""", default=[None, None])
    parser.add_argument("--y_lim", dest="y_lim", nargs=2, help="""Y lims""", default=[None, None])
    parser.add_argument(
        "-v", "--variable", dest="varname", help="""Variable to plot, default = 'velsurf_mag'.""", default="velsurf_mag"
    )

    options = parser.parse_args()
    args = options.FILE

    np.seterr(all="warn")
    aspect_ratio = options.aspect_ratio
    tol = 1e-6
    normalize = options.normalize
    print_mode = options.print_mode
    obs_file = options.obs_file
    out_res = int(options.out_res)
    varname = options.varname
    table_file = options.table_file
    label_params = list(options.label_params.split(","))
    plot_title = options.plot_title
    legend = options.legend
    do_regress = options.do_regress
    make_figures = options.make_figures
    odir = options.odir
    simple_plot = options.simple_plot
    x_lim_min, x_lim_max = options.x_lim
    y_lim_min, y_lim_max = options.y_lim
    ice_density = 910.0
    ice_density_units = "910 kg m-3"
    vol_to_mass = False
    profile_axis_out_units = "km"
    pearson_r_threshold_high = 0.85
    pearson_r_threshold_low = 0.50

    if not os.path.exists(odir):
        os.makedirs(odir)

    if x_lim_min is not None:
        x_lim_min = np.float(x_lim_min)
    if x_lim_max is not None:
        x_lim_max = np.float(x_lim_max)

    if y_lim_min is not None:
        y_lim_min = np.float(y_lim_min)
    if y_lim_max is not None:
        y_lim_max = np.float(y_lim_max)


    if varname in ("velsurf_mag", "velbase_mag", "velsurf_normal"):
        flux_type = "line flux"
        v_o_units = "m yr-1"
        v_o_units_str = "m yr$^\mathregular{{-1}}$"
        v_o_units_str_tex = "m\,yr$^{-1}$"
        v_flux_o_units = "km2 yr-1"
        v_flux_o_units_str = "km$^\mathregular{2}$ yr$^\mathregular{{-1}}$"
        v_flux_o_units_str_tex = "km$^2$\,yr$^{-1}$"
    elif varname in ("flux_mag", "flux_normal"):
        flux_type = "mass flux"
        v_o_units = "km2 yr-1"
        v_o_units_str = "km$^\mathregular{2}$ yr$^\mathregular{{-1}}$"
        v_o_units_str_tex = "km$^2$\,yr$^{-1}$"
        vol_to_mass = True
        v_flux_o_units = "Gt yr-1"
        v_flux_o_units_str = "Gt yr$^\mathregular{{-1}}$"
        v_flux_o_units_str_tex = "Gt\,yr$^{-1}$"
    elif varname in ("thk", "thickness", "land_ice_thickness"):
        flux_type = "area"
        v_o_units = "m"
        v_o_units_str = "m"
        v_o_units_str_tex = "m"
        vol_to_mass = False
        v_flux_o_units = "km2"
        v_flux_o_units_str = "km$^\mathregular{2}$"
        v_flux_o_units_str_tex = "km$^2$"
    elif varname in ("usurf", "surface", "surface_altitude"):
        flux_type = "area"
        v_o_units = "m"
        v_o_units_str = "m"
        v_o_units_str_tex = "m"
        vol_to_mass = False
        v_flux_o_units = "km2"
        v_flux_o_units_str = "km$^\mathregular{2}$"
        v_flux_o_units_str_tex = "km$^2$"
    else:
        print(("variable {} not supported".format(varname)))


    na = len(args)
    shade = 0.15
    colormap = options.colormap
    # FIXME: make option
    cstride = 2
    my_colors = plt.get_cmap(colormap).colors[::cstride]
    my_colors_light = []
    for rgb in my_colors:
        h, l, s = rgb_to_hls(*rgb[0:3])
        l *= 1 + shade
        l = min(0.9, l)
        l = max(0, l)
        my_colors_light.append(hls_to_rgb(h, l, s))

    alpha = 0.75
    dash_style = "o"
    numpoints = 1
    legend_frame_width = 0.25
    markeredgewidth = 0.2
    markeredgecolor = 'k'
    obscolor = "0.4"

    params_dict = {
        "dem": {"abbr": "DEM", "format": "{}"},
        "bed": {"abbr": "bed", "format": "{}"},
        "surface.pdd.factor_ice": {"abbr": "$f_{\mathregular{i}}$", "format": "{:1.0f}"},
        "surface.pdd.factor_snow": {"abbr": "$f_{\mathregular{s}}$", "format": "{:1.0f}"},
        "basal_resistance.pseudo_plastic.q": {"abbr": "$q$", "format": "{:1.2f}"},
        "basal_yield_stress.mohr_coulomb.till_effective_fraction_overburden": {"abbr": "$\\delta$", "format": "{:1.4f}"},
        "stress_balance.sia.enhancement_factor": {"abbr": "$E_{\mathregular{SIA}}$", "format": "{:1.2f}"},
        "stress_balance.ssa.enhancement_factor": {"abbr": "$E_{\mathregular{SSA}}$", "format": "{:1.2f}"},
        "stress_balance.ssa.Glen_exponent": {"abbr": "$n_{\mathregular{SSA}}$", "format": "{:1.2f}"},
        "stress_balance.sia.Glen_exponent": {"abbr": "$n_{\mathregular{SIA}}$", "format": "{:1.2f}"},
        "grid_dx_meters": {"abbr": "ds", "format": "{:.0f}"},
        "flow_law.gpbld.water_frac_observed_limit": {"abbr": "$\omega$", "format": "{:1.2}"},
        "basal_yield_stress.mohr_coulomb.topg_to_phi.phi_min": {"abbr": "$\phi_{\mathregular{min}}$", "format": "{:4.2f}"},
        "basal_yield_stress.mohr_coulomb.topg_to_phi.phi_max": {"abbr": "$\phi_{\mathregular{max}}$", "format": "{:4.2f}"},
        "basal_yield_stress.mohr_coulomb.topg_to_phi.topg_min": {"abbr": "$z_{\mathregular{min}}$", "format": "{:1.0f}"},
        "basal_yield_stress.mohr_coulomb.topg_to_phi.topg_max": {"abbr": "$z_{\mathregular{max}}$", "format": "{:1.0f}"},
    }


    var_long = (
        "velsurf_mag",
        "velbase_mag",
        "velsurf_normal",
        "flux_mag",
        "flux_normal",
        "surface",
        "usurf",
        "surface_altitude" "thk",
        "thickness",
        "land_ice_thickness",
    )
    var_short = (
        "speed",
        "sliding speed",
        "speed",
        "flux",
        "flux",
        "altitude",
        "altitude",
        "altitude",
        "ice thickness",
        "ice thickness",
        "ice thickness",
    )
    var_name_dict = dict(list(zip(var_long, var_short)))

    flow_types = {0: "isbr{\\ae}", 1: "ice-stream", 2: "undefined"}
    glacier_types = {0: "ffmt", 1: "lvmt", 2: "ist", 3: "lt"}


    # Open first file
    filename = args[0]
    print(("  opening NetCDF file %s ..." % filename))
    try:
        nc0 = NC(filename, "r")
    except:
        print(("ERROR:  file '%s' not found or not NetCDF format ... ending ..." % filename))
        import sys

        sys.exit(1)

    # Get profiles from first file
    # All experiments have to contain the same profiles
    # Create flux gates
    profile_names = nc0.variables["profile_name"][:]
    flux_gates = []
    for pos_id, profile_name in enumerate(profile_names):
        profile_axis = nc0.variables["profile_axis"][pos_id]
        profile_axis_units = nc0.variables["profile_axis"].units
        profile_axis_name = nc0.variables["profile_axis"].long_name
        profile_id = int(nc0.variables["profile_id"][pos_id])
        try:
            clon = nc0.variables["clon"][pos_id]
        except:
            clon = 0.0
        try:
            clat = nc0.variables["clat"][pos_id]
        except:
            clat = 0.0
        try:
            flightline = nc0.variables["flightline"][pos_id]
        except:
            flightline = 0
        try:
            glaciertype = nc0.variables["glaciertype"][pos_id]
        except:
            glaciertype = ""
        try:
            flowtype = nc0.variables["flowtype"][pos_id]
        except:
            flowtype = ""
        flux_gate = FluxGate(
            pos_id,
            profile_name,
            profile_id,
            profile_axis,
            profile_axis_units,
            profile_axis_name,
            clon,
            clat,
            flightline,
            glaciertype,
            flowtype,
        )
        flux_gates.append(flux_gate)
    nc0.close()

    # If observations are provided, load observations
    if obs_file:
        obs = ObservationsDataset(obs_file, varname)
        for flux_gate in flux_gates:
            flux_gate.add_observations(obs)

    # Add experiments to flux gates
    for k, filename in enumerate(args):
        # id = re.search("id_(\b0*([1-9][0-9]*|0)\b)", filename).group(1)
        # pid = int(filename.split("id_")[1].split("_")[0])
        id = k
        experiment = ExperimentDataset(id, filename, varname)
        for flux_gate in flux_gates:
            flux_gate.add_experiment(experiment)


    # set the print mode
    lw, pad_inches = ppt.set_mode(print_mode, aspect_ratio=aspect_ratio)

    ne = len(flux_gates[0].experiments)
    ng = len(flux_gates)

    if table_file and obs_file:
        export_latex_table_flux(table_file, flux_gates, label_params)

    # make figure for each flux gate
    for gate in flux_gates:
        if make_figures:
            gate.make_line_plot(label_param_list=label_params)
        else:
            if not gate.has_fluxes:
                gate.calculate_fluxes()
            if gate.has_observations:
                gate.calculate_stats()
