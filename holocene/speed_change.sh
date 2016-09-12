#!/bin/bash

# Copyright (C) 2016 Andy Aschwanden
set -x -e

if [ $# -gt 0 ] ; then
  infile="$1"
fi


for var in surface_mass_balance_average velsurf_mag; do
#for var in surface_mass_balance_average; do
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

exit

outfile=holocene/lgm_modern_$infile
ncks -O -d time,-201,-1 $infile $outfile
ncrename -v nuH[0],nuH0 -v nuH[1],nuH1 $outfile
ncap2 -4 -L 3 -O -s "nu=sqrt(nuH0^2+nuH1^2);" $outfile $outfile

infile2=$outfile
outfile=holocene/hc_modern_$infile
ncks -O -d time,-121,-1 $infile2 $outfile

exit

