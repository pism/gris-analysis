#!/bin/bash

# run in green-hydro/run

set -e -x

# set type from argument 1
pgs=250
if [ -n "$1" ]; then
    pgs=$1
fi

GRID=600
CLIMATE=const
TYPE=ctrl
tl_dir=${GRID}m_${CLIMATE}_${TYPE}
nc_dir=processed
pr_dir=profiles
reg_dir=regional
obs_dir=observed  
fig_dir=figures

out_suffix=sr13
for saryears in "2008-2009"; do
    for region in "greenland"; do
        for var in  "velsurf_normal"; do
             ./flux-gate-analysis.py --legend short --plot_title --colormap Set1 qualitative 4 0 -p 50mm  --label_params 'bed_data_set'  -v ${var} --obs_file $obs_dir/$pr_dir/profiles_${pgs}m_${region}_${saryears}_observed_flux.nc $tl_dir/$pr_dir/profiles_${pgs}m_${region}_g${GRID}m_${CLIMATE}_${TYPE}_e_1.25_ppq_0.6_tefo_0.02_ssa_n_3.25_philow_5.0_hydro_null_100a.nc 5000m_const_searise/$pr_dir/profiles_${pgs}m_${region}_g5000m_const_searise_0_ftt.nc
            for file in *_${var}_profile.pdf; do
                bfile=$(basename $file .pdf)
                mv $file ${bfile}_${pgs}m_${region}_${saryears}_${GRID}m_${out_suffix}.pdf
            done
            mv rmsd_cum_table.tex rmsd_cum_table_${var}_${pgs}m_${GRID}m_${out_suffix}.tex
        done
    done
done
