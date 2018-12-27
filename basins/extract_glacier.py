# Copyright (C) 2016-18 Andy Aschwanden

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import ocgis
import os
from datetime import datetime
from unidecode import unidecode
import fiona

import logging
import logging.handlers

# create logger
logger = logging.getLogger("extract_glacier")
logger.setLevel(logging.DEBUG)

# create file handler which logs even debug messages
fh = logging.handlers.RotatingFileHandler("extract.log")
fh.setLevel(logging.DEBUG)
# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
# create formatter
info_formatter = logging.Formatter("%(message)s")
debug_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s")

# add formatter to ch and fh
ch.setFormatter(info_formatter)
fh.setFormatter(debug_formatter)

# add ch to logger
logger.addHandler(ch)
logger.addHandler(fh)

script_path = os.path.dirname(os.path.realpath(__file__))
default_basin_file = "Greenland_Basins_PS_v1.4.2c.shp"


def extract_glacier(gl_id, gl_name, basin=None):
    """
    Extract glacier using OCGIS
    """

    logger.info("Extracting glacier {id}".format(id=gl_id))
    rd = ocgis.RequestDataset(uri=uri, variable=variable)
    select_geom = [x for x in ocgis.GeomCabinetIterator(path=shape_file) if x["properties"]["id"] == gl_id]
    ## this argument must always come in as a list
    # print(select_geom)
    select_ugid = [select_geom[0]["properties"]["UGID"]]
    ## parameterize the operations to be performed on the target dataset
    ops = ocgis.OcgOperations(
        dataset=rd,
        time_range=time_range,
        geom=shape_file,
        aggregate=False,
        snippet=False,
        select_ugid=select_ugid,
        output_format=output_format,
        output_format_options=output_format_options,
        prefix=prefix,
        dir_output=odir,
    )
    ret = ops.execute()


def calculate_time_series():
    """
    Calculate scalar time series with CDO
    """

    from cdo import Cdo

    cdo = Cdo()

    ifile = os.path.join(odir, prefix, prefix + ".nc")
    scalar_ofile = os.path.join(odir, "scalar", ".".join(["_".join(["ts", prefix]), "nc"]))
    logger.info("Calculating field sum and saving to \n {}".format(scalar_ofile))
    cdo.setattribute(
        'discharge_flux@units="Gt year-1",dMdt@units="Gt year-1"',
        input="-aexpr,dMdt=tendency_of_ice_mass-tendency_of_ice_mass_due_to_flow,discharge_flux=tendency_of_ice_mass_due_to_discharge+tendency_of_ice_mass_due_to_basal_mass_flux -fldsum -selvar,{} {}".format(
            ",".join(mvar for mvar in mvars), ifile
        ),
        output=scalar_ofile,
        overwrite=True,
        options="-L",
    )


# set up the option parser
parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.description = "Extract basins from continental scale files."
parser.add_argument("FILE", nargs=1)
parser.add_argument("--ugid", help="UGID", type=int, default=None)
parser.add_argument("--o_dir", dest="odir", help="output directory", default="../glaciers")
parser.add_argument(
    "--shape_file",
    dest="shape_file",
    help="Path to shape file with basins",
    default=os.path.join(script_path, default_basin_file),
)
parser.add_argument(
    "-v",
    "--variable",
    dest="variable",
    help="Comma-separated list of variables to be extracted. By default, all variables are extracted.",
    default=None,
)
parser.add_argument(
    "--no_extraction", dest="no_extraction", action="store_true", help="Don't extract basins", default=False
)
parser.add_argument(
    "--no_timeseries", dest="no_timeseries", action="store_true", help="Don't calculate time-series", default=False
)
parser.add_argument("--start_date", help="Start date YYYY-MM-DD", default="2008-1-1")
parser.add_argument("--end_date", help="End date YYYY-MM-DD", default="2299-1-1")
options = parser.parse_args()
ugid = options.ugid
no_extraction = options.no_extraction
no_timeseries = options.no_timeseries
time_range = [datetime.strptime(options.start_date, "%Y-%M-%d"), datetime.strptime(options.end_date, "%Y-%M-%d")]


uri = options.FILE[0]
shape_file = options.shape_file
variable = options.variable
if options.variable is not None:
    variable = options.variable.split(",")

odir = options.odir
if not os.path.isdir(odir):
    os.mkdir(odir)
if not os.path.isdir(os.path.join(odir, "scalar")):
    os.mkdir(os.path.join(odir, "scalar"))

ocgis.env.OVERWRITE = True

# Output name
savename = uri[0 : len(uri) - 3]

## set the output format to convert to
output_format = "nc"
output_format_options = {"data_model": "NETCDF4", "variable_kwargs": {"zlib": True, "complevel": 3}}

## we can either subset the data by a geometry from a shapefile, or convert to
## geojson for the entire spatial domain. there are other options here (i.e. a
## bounding box for tiling or a Shapely geometry).
GEOM = shape_file

mvars_dict = {
    "tendency_of_ice_mass": "ice_mass",
    "tendency_of_ice_mass_due_to_flow": "flow_cumulative",
    "tendency_of_ice_mass_due_to_conservation_error": "conservation_error_cumulative",
    "tendency_of_ice_mass_due_to_basal_mass_flux": "basal_mass_flux_cumulative",
    "tendency_of_ice_mass_due_to_surface_mass_flux": "surface_mass_flux_cumulative",
    "tendency_of_ice_mass_due_to_discharge": "discharge_cumulative",
    "saccum": "saccum",
    "smelt": "smelt",
    "srunoff": "srunoff",
}

mvars = list(mvars_dict.keys())
mvars = [
    "ice_mass",
    "tendency_of_ice_mass",
    "tendency_of_ice_mass_due_to_flow",
    "tendency_of_ice_mass_due_to_conservation_error",
    "tendency_of_ice_mass_due_to_basal_mass_flux",
    "tendency_of_ice_mass_due_to_surface_mass_flux",
    "tendency_of_ice_mass_due_to_discharge",
    "surface_accumulation_rate",
    "surface_melt_rate",
    "surface_runoff_rate",
    "sftgif",
]

ds = fiona.open(shape_file, encoding="utf-8")

cvars = ["pism_config"]
variable = mvars

with fiona.open(shape_file, encoding="utf-8") as ds:
    glacier = filter(lambda f: f["properties"]["UGID"] == ugid, ds)

gl_id = glacier[0]["properties"]["id"]
gl_name = glacier[0]["properties"]["Name"]
basin = glacier[0]["properties"]["basin"]
prefix = "b_{basin}_id_{gl_id}_{gl_name}_{savename}".format(
    basin=basin, gl_id=gl_id, gl_name=unidecode(gl_name).replace(" ", "_"), savename=savename
)
if not no_extraction:
    extract_glacier(gl_id, gl_name, basin=basin)
if not no_timeseries:
    calculate_time_series()
