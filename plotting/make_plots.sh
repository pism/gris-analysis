#!/bin/bash

set -x -e 
plot_cmd="python /Volumes/humboldt/data/gris-analysis/plotting/plotting.py"
grid=900
odir=2017_05_vc_tct_250
#${plot_cmd} -b GR --title GRIS -o GRIS -p twocol -a 0.35 ${odir}/scalar/ts_gris_g${grid}m_warming_v3a_no_bath_sia_e_1.25_lapse_6_tm_1_bed_deformation_off_calving_vonmises_calving_threshold_250_0_200.nc


discharge_flux_all_basins() {
    ${plot_cmd} --plot basin_discharge --time_bounds 2008 2200 -o gris_all -p twocol -a 0.30 ${odir}/spatial/ctrl/b_*_ex_gris_g${grid}m_warming_v3a_no_bath_sia_e_1.25_lapse_6_tm_1_bed_deformation_off_calving_vonmises_calving_threshold_250/scalar_fldsum_b_*_ex_gris_g${grid}m_warming_v3a_no_bath_sia_e_1.25_lapse_6_tm_1_bed_deformation_off_calving_vonmises_calving_threshold_250.nc ${odir}/scalar/ts_gris_g${grid}m_warming_v3a_no_bath_sia_e_1.25_lapse_6_tm_1_bed_deformation_off_calving_vonmises_calving_threshold_250_0_200.nc

    ${plot_cmd} --plot basin_discharge --time_bounds 2008 2200 -o all -p twocol -a 0.30 ${odir}/spatial/ctrl/b_*_ex_gris_g${grid}m_warming_v3a_no_bath_sia_e_1.25_lapse_6_tm_1_bed_deformation_off_calving_vonmises_calving_threshold_250/scalar_fldsum_b_*_ex_gris_g${grid}m_warming_v3a_no_bath_sia_e_1.25_lapse_6_tm_1_bed_deformation_off_calving_vonmises_calving_threshold_250.nc

    ${plot_cmd} --plot basin_discharge --time_bounds 2008 2200 -o runmean_10yr_all -p twocol -a 0.30 ${odir}/spatial/ctrl/b_*_ex_gris_g${grid}m_warming_v3a_no_bath_sia_e_1.25_lapse_6_tm_1_bed_deformation_off_calving_vonmises_calving_threshold_250/runmean_10yr_b_*_ex_gris_g${grid}m_warming_v3a_no_bath_sia_e_1.25_lapse_6_tm_1_bed_deformation_off_calving_vonmises_calving_threshold_250.nc

    ${plot_cmd} --plot basin_discharge --time_bounds 2008 2200 -o abs_anomaly_all -p twocol -a 0.30 ${odir}/spatial/ctrl/b_*_ex_gris_g${grid}m_warming_v3a_no_bath_sia_e_1.25_lapse_6_tm_1_bed_deformation_off_calving_vonmises_calving_threshold_250/abs_anomaly_runmean_10yr_b_*_ex_gris_g${grid}m_warming_v3a_no_bath_sia_e_1.25_lapse_6_tm_1_bed_deformation_off_calving_vonmises_calving_threshold_250.nc

    ${plot_cmd} --plot rel_basin_discharge --time_bounds 2008 2200 -o rel_anomaly_all -p twocol -a 0.30 ${odir}/spatial/ctrl/b_*_ex_gris_g${grid}m_warming_v3a_no_bath_sia_e_1.25_lapse_6_tm_1_bed_deformation_off_calving_vonmises_calving_threshold_250/rel_anomaly_runmean_10yr_b_*_ex_gris_g${grid}m_warming_v3a_no_bath_sia_e_1.25_lapse_6_tm_1_bed_deformation_off_calving_vonmises_calving_threshold_250.nc

}



fluxes_per_basin() {
    for basin in "CW" "NE" "NO" "NW" "SE" "SW"; do
        ${plot_cmd} -b ${basin} --title ${basin} -o ${basin} -p twocol -a 0.30 ${odir}/spatial/ctrl/b_${basin}_ex_gris_g${grid}m_warming_v3a_no_bath_sia_e_1.25_lapse_6_tm_1_bed_deformation_off_calving_vonmises_calving_threshold_250/scalar_fldsum_b_${basin}_ex_gris_g${grid}m_warming_v3a_no_bath_sia_e_1.25_lapse_6_tm_1_bed_deformation_off_calving_vonmises_calving_threshold_250.nc
        ${plot_cmd} -b ${basin} --title ${basin} -o runmean_10yr_${basin} -p twocol -a 0.30 ${odir}/spatial/ctrl/b_${basin}_ex_gris_g${grid}m_warming_v3a_no_bath_sia_e_1.25_lapse_6_tm_1_bed_deformation_off_calving_vonmises_calving_threshold_250/runmean_10yr_b_${basin}_ex_gris_g${grid}m_warming_v3a_no_bath_sia_e_1.25_lapse_6_tm_1_bed_deformation_off_calving_vonmises_calving_threshold_250.nc
        ${plot_cmd} -b ${basin} --title ${basin} -o anomaly_${basin} -p twocol -a 0.30 ${odir}/spatial/ctrl/b_${basin}_ex_gris_g${grid}m_warming_v3a_no_bath_sia_e_1.25_lapse_6_tm_1_bed_deformation_off_calving_vonmises_calving_threshold_250/anomaly_runmean_10yr_b_${basin}_ex_gris_g${grid}m_warming_v3a_no_bath_sia_e_1.25_lapse_6_tm_1_bed_deformation_off_calving_vonmises_calving_threshold_250.nc
    done
}

discharge_flux_all_basins
#fluxes_per_basin
