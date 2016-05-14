#!/bin/bash

# run in green-hydro/run

set -e -x

GRID=1500
# set grid spacing from argument 1
if [ -n "$1" ]; then
    GRID=$1
fi
CLIMATE=const
# set climate from argument 2
if [ -n "$2" ]; then
    CLIMATE=$2
fi
TYPE=ctrl
# set type from argument 3
if [ -n "$3" ]; then
    TYPE=$3
fi

tl_dir=${GRID}m_${CLIMATE}_${TYPE}
nc_dir=processed
ct_dir=speed_contours

if [ ! -d ${tl_dir}/${ct_dir} ]; then
    mkdir -p ${tl_dir}/${ct_dir}
fi

# Extract speed contours from simulations

cd ${tl_dir}/${nc_dir}
for file in g${GRID}m_${CLIMATE}_${TYPE}_*; do
    fbname=$(basename "$file" .nc)
    rm -f ../${ct_dir}/${fbname}_speed_contours.*
    gdal_contour -fl 20 200 2000 -a speed NETCDF:${file}:velsurf_mag ../${ct_dir}/${fbname}_speed_contours.shp
    ogr2ogr -s_srs epsg:3413 -t_srs epsg:4326 -overwrite ../${ct_dir}/${fbname}_speed_contours_epsg4623.shp ../${ct_dir}/${fbname}_speed_contours.shp
done
cd ../../
