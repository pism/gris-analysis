#!/bin/bash

# run in green-hydro/run
#!/bin/bash

# Copyright (C) 2014-2015 Andy Aschwanden

# script used in Aschwanden, Fahnestock, and Truffer (2016):
# generates Figs. 6-8 and Table 4 in the SI
# run in green-hydro/run

set -e -x


CLIMATE=const
TYPE=ctrl

pgs=250
if [ -n "$1" ]; then
    pgs=$1
fi


ver=1.1
CLIMATE=const
TYPE=ctrl

nc_dir=processed
pr_dir=profiles
reg_dir=regional
obs_dir=observed  
fig_dir=figures

out_suffix="600m-vs-4500m"
for saryears in "2008-2009"; do

    for region in "greenland"; do
        for var in  "velsurf_normal"; do

            GRID=600
            tl_dir=${GRID}m_${CLIMATE}_${TYPE}
            ./flux-gate-analysis.py --legend short --plot_title --colormap Set1 qualitative 4 0 -p 50mm  --label_params 'bed_data_set,grid_dx_meters'  -v ${var} --obs_file $obs_dir/$pr_dir/profiles_${pgs}m_${region}_${saryears}_observed_flux.nc $tl_dir/$pr_dir/profiles_${pgs}m_${region}_g${GRID}m_${CLIMATE}_${TYPE}_e_1.25_ppq_0.6_tefo_0.02_ssa_n_3.25_philow_5.0_hydro_null_100a.nc 4500m_${CLIMATE}_ctrl/$pr_dir/profiles_${pgs}m_${region}_g4500m_${CLIMATE}_ctrl_e_1.25_ppq_0.6_tefo_0.02_ssa_n_3.25_philow_5.0_hydro_null_100a.nc 600m_${CLIMATE}_old_bed/$pr_dir/profiles_${pgs}m_${region}_g600m_${CLIMATE}_old_bed_e_1.25_ppq_0.6_tefo_0.02_ssa_n_3.25_philow_5.0_hydro_null_100a.nc 4500m_${CLIMATE}_old_bed/$pr_dir/profiles_${pgs}m_${region}_g4500m_${CLIMATE}_old_bed_e_1.25_ppq_0.6_tefo_0.02_ssa_n_3.25_philow_5.0_hydro_null_100a.nc
            for file in *_${var}_profile.pdf; do
                bfile=$(basename $file .pdf)
                mv $file ${bfile}_${pgs}m_${region}_${saryears}_${out_suffix}.pdf
            done
            for file in *_rmsd_${var}.tex; do
                bfile=$(basename $file .tex)
                mv $file ${bfile}_${pgs}m_${region}_${saryears}_${out_suffix}.tex
            done
            mv pearson_r_cum_table.tex pearson_r_cum_table_${out_suffix}.tex

            mv rmsd_cum_table.tex rmsd_cum_table_${var}_${pgs}m_${out_suffix}.tex
            mv statistics.shp statistics_${var}_${pgs}m_${region}_${saryears}_${out_suffix}.shp
            mv statistics.dbf statistics_${var}_${pgs}m_${region}_${saryears}_${out_suffix}.dbf
            mv statistics.prj statistics_${var}_${pgs}m_${region}_${saryears}_${out_suffix}.prj
            mv statistics.shx statistics_${var}_${pgs}m_${region}_${saryears}_${out_suffix}.shx

        done
    done
done
