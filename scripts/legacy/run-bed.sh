#!/bin/bash

# run in green-hydro/run

set -e -x


CLIMATE=const
TYPE=ctrl

pgs=250
if [ -n "$1" ]; then
    pgs=$1
fi

ver=1.1
GRID=600
CLIMATE=const
TYPE=ctrl
tl_dir=${GRID}m_${CLIMATE}_${TYPE}
nc_dir=processed
pr_dir=profiles
reg_dir=regional
obs_dir=observed  
fig_dir=figures

for saryears in "2008-2009"; do

    for region in "greenland"; do
        for var in  "velsurf_normal"; do
             ./flux-gate-analysis.py --legend short --plot_title --colormap Set1 qualitative 4 0 -p 50mm  --label_params 'bed_data_set'  -v ${var} --obs_file $obs_dir/$pr_dir/profiles_${pgs}m_${region}_${saryears}_observed_flux_v${ver}.nc $tl_dir/$pr_dir/profiles_${pgs}m_${region}_g${GRID}m_${CLIMATE}_${TYPE}_e_1.25_ppq_0.6_tefo_0.02_ssa_n_3.25_philow_5.0_hydro_null_100a.nc $tl_dir/$pr_dir/profiles_250m_greenland_g600m_phi.nc
            for file in *_${var}_profile.pdf; do
                bfile=$(basename $file .pdf)
                mv $file ${bfile}_${pgs}m_${region}_${saryears}_${GRID}m_bed.pdf
            done
            mv rmsd_cum_table.tex rmsd_cum_table_${var}_${pgs}m_${GRID}m_bed.tex
        done
    done
done
