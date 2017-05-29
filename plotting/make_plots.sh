#!/bin/bash

set -x -e 
plot_cmd="python /Volumes/humboldt/data/gris-analysis/plotting/plot_fluxes.py"
grid=900
odir=2017_05_vc_tct_250
${plot_cmd} --title GRIS -o GRIS -p twocol -a 0.35 ${odir}/scalar/ts_gris_g${grid}m_warming_v3a_no_bath_sia_e_1.25_lapse_6_tm_1_bed_deformation_off_calving_vonmises_calving_threshold_250_0_200.nc

for basin in "CW" "NE" "NW" "NO" "SE" "SW"; do
    ${plot_cmd} --title ${basin} -o ${basin} -p twocol -a 0.35 ${odir}/spatial/ctrl/basin_${basin}_ex_gris_g${grid}m_warming_v3a_no_bath_sia_e_1.25_lapse_6_tm_1_bed_deformation_off_calving_vonmises_calving_threshold_250/scalar_fldsum_basin_${basin}_ex_gris_g${grid}m_warming_v3a_no_bath_sia_e_1.25_lapse_6_tm_1_bed_deformation_off_calving_vonmises_calving_threshold_250.nc
done 
