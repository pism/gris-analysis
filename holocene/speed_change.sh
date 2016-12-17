#!/bin/bash

# Copyright (C) 2016 Andy Aschwanden
set -x -e

if [ $# -gt 0 ] ; then
  infile="$1"
fi

outfile=holocene/humboldt_lgm_modern_$infile
# ncks -O -d time,-201,-1 -d x,-1254650.,-95000. -d y,-1450000.,-527600. $infile $outfile

infile2=$outfile
outfile=holocene/humboldt_hc_modern_$infile
# ncks -O -d time,-121,-1 $infile2 $outfile

infile2=$outfile

outfile=holocene/humboldt_12ka_$infile
ncks -O -d time,0 $infile2 $outfile

filename=$(basename "$outfile")
gdal_contour -fl 10 NETCDF:$outfile:thk ${filename}_10m_thk.shp

outfile=holocene/humboldt_8ka_$infile
ncks -O -d time,-41 $infile2 $outfile

filename=$(basename "$outfile")
gdal_contour -fl 10 NETCDF:$outfile:thk ${filename}_10m_thk.shp


exit

outfile=holocene/lgm_modern_$infile
ncks -O -d time,-201,-1 $infile $outfile
ncrename -v nuH[0],nuH0 -v nuH[1],nuH1 $outfile
ncap2 -4 -L 3 -O -s "nu=sqrt(nuH0^2+nuH1^2);" $outfile $outfile

infile2=$outfile
outfile=holocene/hc_modern_$infile
ncks -O -d time,-121,-1 $infile2 $outfile

exit

for var in surface_mass_balance_average velsurf_mag; do
    outfile=holocene/${var}_modern_9ka_$infile
    outfilerel=holocene/rel_${var}_modern_9ka_$infile
    ncra -O -v x,y,mapping,${var} -d time,-91,-1 $infile tmp_9ka_$infile
    ncks -O -v x,y,mapping,${var} -d time,-1 $infile tmp_modern_$infile
    ncdiff -O tmp_modern_$infile tmp_9ka_$infile $outfile
    ncks -A -v mask -d time,-1 $infile $outfile
    ncap2 -O -s "where(mask!=2) ${var}=-2e9;" $outfile $outfile
    ncatted -a valid_min,${var},d,, $outfile
    cdo div -selvar,$var $outfile -selvar,$var tmp_9ka_$infile $outfilerel
    ncks -A -v x,y,mapping $infile $outfilerel
    ncatted -a units,$var,d,, -a grid_mapping,$var,o,c,"mapping" $outfilerel
    rm tmp_9ka_$infile tmp_modern_$infile
done



