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


GRID=600
tl_dir=${GRID}m_${CLIMATE}_${TYPE}
for region in "greenland"; do
    for saryears in "2008-2009"; do
        for var in "velsurf_normal"; do
            ./flux-gate-analysis.py --legend short --colormap Set1 qualitative 4 0 -p 50mm --label_params 'ssa_Glen_exponent,pseudo_plastic_q,bed_data_set' -v ${var} --plot_title --obs_file $obs_dir/$pr_dir/profiles_${pgs}m_${region}_${saryears}_observed_flux_v${ver}.nc $tl_dir/$pr_dir/profiles_${pgs}m_${region}_g${GRID}m_${CLIMATE}_${TYPE}_e_1.25_ppq_0.6_tefo_0.02_ssa_n_3.25_philow_5.0_hydro_null_100a.nc $tl_dir/$pr_dir/profiles_${pgs}m_${region}_g${GRID}m_${CLIMATE}_${TYPE}_e_1.25_ppq_0.6_tefo_0.02_ssa_n_3.25_philow_const_30_hydro_null_100a.nc
            for file in *_${var}_profile.pdf; do
                bfile=$(basename $file .pdf)
                mv $file ${bfile}_${pgs}m_${region}_${saryears}_const_tillphi.pdf
            done
            for file in *_pearson_r_${var}.tex; do
                bfile=$(basename $file .tex)
                mv $file ${bfile}_${pgs}m_${region}_${saryears}_const_tillphi.tex
            done
            mv rmsd_cum_table.tex rmsd_cum_table_${var}_${pgs}m_${GRID}m_const_tillphi.tex
        done
    done
done
