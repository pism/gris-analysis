#!/bin/bash

# run in green-hydro/run

set -e -x

# set type from argument 1
pgs=250
if [ -n "$1" ]; then
    pgs=$1
fi

nc_dir=processed
pr_dir=profiles
reg_dir=regional
obs_dir=observed  
fig_dir=figures

ver=1.1
CLIMATE=const
TYPE=ctrl

GRID=1500
tl_dir=${GRID}m_${CLIMATE}_${TYPE}

out_suffix=all
for region in "greenland"; do
    for saryears in "2008-2009"; do
        for var in "velsurf_normal"; do
            ./flux-gate-analysis.py --plot_title -p 72mm --legend short  --export_table_file ${var}-gate-table-all-${pgs}m.tex --label_params 'grid_dx_meters,bed_data_set,ssa_Glen_exponent,pseudo_plastic_q'  -v ${var} --obs_file $obs_dir/$pr_dir/profiles_${pgs}m_${region}_${saryears}_observed_flux.nc  $tl_dir/$pr_dir/profiles_${pgs}m_${region}_g${GRID}m_${CLIMATE}_${TYPE}_e_1.25_ppq_0.*_tefo_0.02_ssa_n_*_hydro_null_100a.nc 600m_${CLIMATE}_${TYPE}/$pr_dir/profiles_${pgs}m_${region}_g600m_${CLIMATE}_${TYPE}_e_1.25_ppq_0.6_tefo_0.02_ssa_n_3.25_philow_5.0_hydro_null_100a.nc 1200m_${CLIMATE}_${TYPE}/${pr_dir}/profiles_${pgs}m_${region}_g1200m_${CLIMATE}_${TYPE}_e_1.25_ppq_0.6_tefo_0.02_ssa_n_3.25_philow_5.0_hydro_null_100a.nc 1800m_${CLIMATE}_${TYPE}/${pr_dir}/profiles_${pgs}m_${region}_g1800m_${CLIMATE}_${TYPE}_e_1.25_ppq_0.6_tefo_0.02_ssa_n_3.25_philow_5.0_hydro_null_100a.nc 3600m_${CLIMATE}_${TYPE}/${pr_dir}/profiles_${pgs}m_${region}_g3600m_${CLIMATE}_${TYPE}_e_1.25_ppq_0.6_tefo_0.02_ssa_n_3.25_philow_5.0_hydro_null_100a.nc 4500m_${CLIMATE}_${TYPE}/${pr_dir}/profiles_${pgs}m_${region}_g4500m_${CLIMATE}_${TYPE}_e_1.25_ppq_0.6_tefo_0.02_ssa_n_3.25_philow_5.0_hydro_null_100a.nc 600m_${CLIMATE}_old_bed/$pr_dir/profiles_${pgs}m_${region}_g600m_${CLIMATE}_old_bed_e_1.25_ppq_0.6_tefo_0.02_ssa_n_3.25_philow_5.0_hydro_null_100a.nc 4500m_${CLIMATE}_old_bed/$pr_dir/profiles_${pgs}m_${region}_g4500m_${CLIMATE}_old_bed_e_1.25_ppq_0.6_tefo_0.02_ssa_n_3.25_philow_5.0_hydro_null_100a.nc

            mv experiment_table.tex experiment_table_v${ver}_${var}_${pgs}m_${out_suffix}.tex

        done
    done
done

