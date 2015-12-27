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

ver=2
CLIMATE=const
TYPE=1985_v${ver}

GRID=1200
tl_dir=relax_${GRID}m_${CLIMATE}_${TYPE}

out_suffix=${GRID}m_calib
for region in "jakobshavn"; do
    for saryears in "1985"; do
        for var in "velsurf_normal"; do
            ./flux-gate-analysis.py --plot_title -p 72mm --legend long  --label_params 'ocean_forcing_type,eigen_calving_K,thickness_calving_threshold'  -v ${var} --obs_file $obs_dir/$pr_dir/profiles_${pgs}m_${region}_${saryears}_observed_flux_v${ver}_1985.nc  $tl_dir/$pr_dir/profiles_${pgs}m_${region}_g${GRID}m_${CLIMATE}_${TYPE}_*2.nc
            for file in pearson_r_experiment_*_${var}.pdf; do
                bfile=$(basename $file .pdf)
                mv $file ${bfile}_v${ver}_${pgs}m_${region}_${saryears}_${out_suffix}.pdf
            done
            for file in *_${var}.tex; do
                bfile=$(basename $file .tex)
                mv $file ${bfile}_v${ver}_${pgs}m_${region}_${saryears}_${out_suffix}.tex
            done
            mv experiment_table.tex experiment_table_v${ver}_${var}_${pgs}m_${out_suffix}.tex
            mv rmsd_cum_table.tex rmsd_cum_table_v${ver}_${var}_${pgs}m_${out_suffix}.tex
            mv pearson_r_cum_table.tex pearson_r_cum_table_v${ver}_${var}_${pgs}m_${out_suffix}.tex
        done
    done
done
