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

CLIMATE=const
TYPE=ctrl

GRID=600
tl_dir=${GRID}m_${CLIMATE}_${TYPE}

out_suffix=e_1.5
for region in "greenland"; do
    for saryears in "2008-2009"; do
        for var in "velsurf_normal"; do
            ./flux-gate-analysis.py --legend long --colormap Set1 qualitative 4 0 -p 50mm --label_params 'sia_enhancement_factor' -v ${var} --plot_title --obs_file $obs_dir/$pr_dir/profiles_${pgs}m_${region}_${saryears}_observed_flux.nc $tl_dir/$pr_dir/profiles_${pgs}m_${region}_g${GRID}m_${CLIMATE}_${TYPE}_e_1.25_ppq_0.6_tefo_0.02_ssa_n_3.25_philow_5.0_hydro_null_100a.nc  600m_const_ctrl/profiles/profiles_250m_g600m_const_ctrl_e_1.5_ppq_0.6_tefo_0.02_ssa_n_3.25_philow_5_hydro_null_100a_s.nc
            for file in *_${var}_profile.pdf; do
                bfile=$(basename $file .pdf)
                mv $file ${bfile}_${pgs}m_${region}_${saryears}_${out_suffix}.pdf
            done
            for file in pearson_r_experiment_*_${var}.pdf; do
                bfile=$(basename $file .pdf)
                mv $file ${bfile}_${pgs}m_${region}_${saryears}_${out_suffix}.pdf
            done
            for file in *_pearson_r_${var}.tex; do
                bfile=$(basename $file .tex)
                mv $file ${bfile}_${pgs}m_${region}_${saryears}_${out_suffix}.tex
            done
        done
    done
done
