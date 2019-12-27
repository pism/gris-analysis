# Copyright (C) 2016-18 Andy Aschwanden
import faulthandler

faulthandler.enable()

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from cdo import Cdo
from datetime import datetime
import fiona
from functools import partial
import logging
import logging.handlers
import multiprocessing as mp
import numpy as np
import ocgis
import os
from unidecode import unidecode

cdo = Cdo()

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
default_basin_file = "Greenland_Basins_PS_v1.4.2ext_TW.shp"


def extract_glacier_by_ugid(glacier, ugid, uri, shape_file, variable, metadata):
    """
    Extract glacier using OCGIS
    """

    output_dir = metadata["output_dir"]
    output_format = metadata["output_format"]
    output_format_options = metadata["output_format_options"]
    prefix = metadata["prefix_string"]
    time_range = metadata["time_range"]

    logger.info("Extracting glacier {} with UGID {}".format(glacier, ugid))
    rd = ocgis.RequestDataset(uri=uri, variable=variable)
    select_ugid = [
        [x for x in ocgis.GeomCabinetIterator(path=shape_file) if x["properties"]["UGID"] == ugid][0]["properties"][
            "UGID"
        ]
    ]
    ## parameterize the operations to be performed on the target dataset
    ops = ocgis.OcgOperations(
        dataset=rd,
        time_range=time_range,
        geom=shape_file,
        # aggregate=True,
        snippet=False,
        # calc=[{"func": "mean", "name": "m"}],
        # calc_grouping=["all"],
        select_ugid=ugid,
        output_format=output_format,
        output_format_options=output_format_options,
        prefix=prefix,
        dir_output=output_dir,
    )
    ret = ops.execute()


def calculate_field(cdo_operator, infile, outfile):

    """
    Calculate scalar time series with CDO
    """

    logger.info("Calculating field sum and saving to \n {}".format(outfile))
    cdo_operator(input="-setctomiss,0 {}".format(infile), output=outfile, overwrite=True, options="-O -b F32")


def calculate_field_timmean(cdo_operator, infile, outfile):

    """
    Calculate scalar time series with CDO
    """

    logger.info("Calculating field sum and saving to \n {}".format(outfile))
    cdo_operator(input="-timmean -setctomiss,0 {}".format(infile), output=outfile, overwrite=True, options="-O -b F32")


def calculate_field_timcumsum(cdo_operator, infile, outfile):

    """
    Calculate scalar time series with CDO
    """

    logger.info("Calculating field sum and saving to \n {}".format(outfile))
    cdo_operator(
        input="-timcumsum -setctomiss,0 {}".format(infile), output=outfile, overwrite=True, options="-O -b F32"
    )


def extract(ugid, metadata):

    idx = np.where(np.asarray(metadata["ugids"]) == ugid)[0][0]
    gl_name = metadata["names"][idx]
    odir = metadata["output_dir"]
    savename = metadata["savename"]
    shape_file = metadata["shape_file"]
    uri = metadata["uri"]
    variable = metadata["variable"]

    prefix = "ugid_{ugid}_{gl_name}_{savename}".format(
        ugid=ugid, gl_name=unidecode(gl_name).replace(" ", "_"), savename=savename
    )
    metadata["prefix_string"] = prefix
    no_extraction = metadata["no_extraction"]
    if not no_extraction:
        extract_glacier_by_ugid(gl_name, ugid, uri, shape_file, variable, metadata)
    no_scalars = metadata["no_scalars"]
    if not no_scalars:

        infile = os.path.join(odir, prefix, prefix + ".nc")
        infile = infile.replace("(", "\(").replace(")", "\)")
        cdo_operator = cdo.fldsum
        outfile = os.path.join(odir, "scalar", ".".join(["_".join(["fldsum", prefix]), "nc"]))
        outfile = outfile.replace("(", "").replace(")", "")
        calculate_field(cdo_operator, infile, outfile)
        outfile = os.path.join(odir, "scalar", ".".join(["_".join(["fldsum_timcumsum", prefix]), "nc"]))
        calculate_field_timcumsum(cdo_operator, infile, outfile)
        # outfile = os.path.join(odir, "scalar", ".".join(["_".join(["fldsum_timmean", prefix]), "nc"]))
        # calculate_field_timmean(cdo_operator, infile, outfile)
        # cdo_operator = cdo.fldmean
        # outfile = os.path.join(odir, "scalar", ".".join(["_".join(["fldmean", prefix]), "nc"]))
        # calculate_field(cdo_operator, infile, outfile)
        # outfile = os.path.join(odir, "scalar", ".".join(["_".join(["fldmean_timmean", prefix]), "nc"]))
        # calculate_field_timmean(cdo_operator, infile, outfile)


if __name__ == "__main__":

    __spec__ = None

    # set up the option parser

    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.description = "Extract glaciers from continental scale files."
    parser.add_argument("FILE", nargs=1)
    parser.add_argument("--ugid", help="UGID", default=None)
    parser.add_argument("--o_dir", dest="odir", help="output directory", default="../glaciers")
    parser.add_argument(
        "--shape_file",
        dest="shape_file",
        help="Path to shape file with basins",
        default=os.path.join(script_path, default_basin_file),
    )
    parser.add_argument(
        "-n",
        "--n_procs",
        dest="n_procs",
        type=int,
        help="""number of cores/processors. default=4. Only used if --ugid all""",
        default=4,
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
        "--no_scalars",
        dest="no_scalars",
        action="store_true",
        help="Don't scalar time-series by using 'fld*' operators",
        default=False,
    )
    parser.add_argument("--start_date", help="Start date YYYY-MM-DD", default="2008-1-1")
    parser.add_argument("--end_date", help="End date YYYY-MM-DD", default="2299-1-1")
    options = parser.parse_args()
    ugid = options.ugid
    time_range = [datetime.strptime(options.start_date, "%Y-%M-%d"), datetime.strptime(options.end_date, "%Y-%M-%d")]
    n_procs = options.n_procs
    no_extraction = options.no_extraction
    no_scalars = options.no_scalars
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
    # output_format_options = {}

    ## we can either subset the data by a geometry from a shapefile, or convert to
    ## geojson for the entire spatial domain. there are other options here (i.e. a
    ## bounding box for tiling or a Shapely geometry).
    GEOM = shape_file

    with fiona.open(shape_file, encoding="utf-8") as ds:

        glacier_names = []
        glacier_ugids = []
        for item in ds.items():
            glacier_names.append(item[1]["properties"]["Name"])
            glacier_ugids.append(item[1]["properties"]["UGID"])

    metadata = {
        "names": glacier_names,
        "ugids": glacier_ugids,
        "savename": savename,
        "no_extraction": no_extraction,
        "no_scalars": no_scalars,
        "output_dir": odir,
        "output_format": output_format,
        "output_format_options": output_format_options,
        "shape_file": shape_file,
        "time_range": time_range,
        "uri": uri,
        "variable": variable,
    }

    if ugid == "all":

        with mp.Pool(n_procs) as pool:
            mp.set_start_method("forkserver", force=True)
            pool.map(partial(extract, metadata=metadata), glacier_ugids)
            pool.close()

    else:
        extract(int(ugid), metadata=metadata)
