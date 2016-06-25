#!/bin/bash

# Copyright (C) 2016 Andy Aschwanden

if [ $# -gt 0 ] ; then  # if user says "paramspawn.sh 8" then NN = 8
  infile="$1"
fi

outfile=velsurf_mag_modern_9ka_$infile
ncra -O -v x,y,mapping,velsurf_mag -d time,-91,-1 $infile tmp_9ka.nc
ncks -O -v x,y,mapping,velsurf_mag -d time,-1 $infile tmp_modern.nc
ncdiff -O tmp_modern.nc tmp_9ka.nc $outfile
ncks -A -v mask -d time,-1 $infile $outfile
ncap2 -O -s "where(mask!=2) velsurf_mag=-2e9;" $outfile $outfile
ncatted -a valid_min,velsurf_mag,d,, $outfile
rm tmp_9ka.nc tmp_modern.nc
