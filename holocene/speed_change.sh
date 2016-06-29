#!/bin/bash

# Copyright (C) 2016 Andy Aschwanden

if [ $# -gt 0 ] ; then
  infile="$1"
fi

outfile=holocene/lgm_modern_$infile
ncks -O -d time,-201,-1 $infile $outfile
ncrename -v nuH[0],nuH0 -v nuH[1],nuH1 $outfile
ncap2 -4 -L 3 -O -s "nu=sqrt(nuH0^2+nuH1^2);" $outfile $outfile

exit

outfile=holocene/hc_modern_$infile
ncks -O -d time,-121,-1 $infile $outfile
ncrename -v nuH[0],nuH0 -v nuH[1],nuH1 $outfile
ncap2 -4 -L 3 -O -s "nu=sqrt(nuH0^2+nuH1^2);" $outfile $outfile

outfile=holocene/velsurf_mag_modern_9ka_$infile
ncra -O -v x,y,mapping,velsurf_mag -d time,-91,-1 $infile tmp_9ka_$infile
ncks -O -v x,y,mapping,velsurf_mag -d time,-1 $infile tmp_modern_$infile
ncdiff -O tmp_modern_$infile tmp_9ka_$infilec $outfile
ncks -A -v mask -d time,-1 $infile $outfile
ncap2 -O -s "where(mask!=2) velsurf_mag=-2e9;" $outfile $outfile
ncatted -a valid_min,velsurf_mag,d,, $outfile
rm tmp_9ka_$infile tmp_modern_$infile
