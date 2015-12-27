#!/bin/bash

# Copyright (C) 2014-2015 Andy Aschwanden

# script used in Aschwanden, Fahnestock, and Truffer (2016):
# generates Fig. 2 in the main manuscript and Figs. 3-5 in the SI
# run in green-hydro/run

set -e -x

ver=1.1
CLIMATE=const
TYPE=ctrl

pgs=250
if [ -n "$1" ]; then
    pgs=$1
fi


tl_dir=${GRID}m_${CLIMATE}_${TYPE}
nc_dir=processed
pr_dir=profiles
reg_dir=regional
obs_dir=observed  
fig_dir=figures

out_suffix=grid_mo14_flow_v${ver}
for saryears in "2008-2009"; do
    for region in "greenland"; do
        for var in  "velsurf_normal"; do
            # without legend
          ./flux-gate-analysis.py --legend none --plot_title -p 72mm -v $var --colormap Greens Sequential 5 0 --do_regress --label_params 'grid_dx_meters' --obs_file $obs_dir/$pr_dir/profiles_${pgs}m_${region}_${saryears}_observed_flux_v${ver}.nc 600m_${CLIMATE}_${TYPE}/$pr_dir/profiles_${pgs}m_${region}_g600m_${CLIMATE}_${TYPE}_e_1.25_ppq_0.6_tefo_0.02_ssa_n_3.25_philow_5.0_hydro_null_100a.nc 1500m_${CLIMATE}_${TYPE}/${pr_dir}/profiles_${pgs}m_${region}_g1500m_${CLIMATE}_${TYPE}_e_1.25_ppq_0.6_tefo_0.02_ssa_n_3.25_philow_5.0_hydro_null_100a.nc 3600m_${CLIMATE}_${TYPE}/${pr_dir}/profiles_${pgs}m_${region}_g3600m_${CLIMATE}_${TYPE}_e_1.25_ppq_0.6_tefo_0.02_ssa_n_3.25_philow_5.0_hydro_null_100a.nc 4500m_${CLIMATE}_${TYPE}/${pr_dir}/profiles_${pgs}m_${region}_g4500m_${CLIMATE}_${TYPE}_e_1.25_ppq_0.6_tefo_0.02_ssa_n_3.25_philow_5.0_hydro_null_100a.nc
            for file in *_${var}_profile.pdf; do
                bfile=$(basename $file .pdf)
                mv $file ${bfile}_${pgs}m_${region}_${saryears}_${out_suffix}_sel_green_noleg.pdf
            done
            ./flux-gate-analysis.py --legend none --plot_title -p 72mm -v $var --colormap Blues Sequential 5 0 --do_regress --label_params 'grid_dx_meters' --obs_file $obs_dir/$pr_dir/profiles_${pgs}m_${region}_${saryears}_observed_flux_v${ver}.nc 600m_${CLIMATE}_${TYPE}/$pr_dir/profiles_${pgs}m_${region}_g600m_${CLIMATE}_${TYPE}_e_1.25_ppq_0.6_tefo_0.02_ssa_n_3.25_philow_5.0_hydro_null_100a.nc 1500m_${CLIMATE}_${TYPE}/${pr_dir}/profiles_${pgs}m_${region}_g1500m_${CLIMATE}_${TYPE}_e_1.25_ppq_0.6_tefo_0.02_ssa_n_3.25_philow_5.0_hydro_null_100a.nc 3600m_${CLIMATE}_${TYPE}/${pr_dir}/profiles_${pgs}m_${region}_g3600m_${CLIMATE}_${TYPE}_e_1.25_ppq_0.6_tefo_0.02_ssa_n_3.25_philow_5.0_hydro_null_100a.nc 4500m_${CLIMATE}_${TYPE}/${pr_dir}/profiles_${pgs}m_${region}_g4500m_${CLIMATE}_${TYPE}_e_1.25_ppq_0.6_tefo_0.02_ssa_n_3.25_philow_5.0_hydro_null_100a.nc
            for file in *_${var}_profile.pdf; do
                bfile=$(basename $file .pdf)
                mv $file ${bfile}_${pgs}m_${region}_${saryears}_${out_suffix}_sel_blue_no_leg.pdf
            done
            ./flux-gate-analysis.py --legend none --plot_title -p 72mm -v $var --colormap Reds Sequential 5 0 --do_regress --label_params 'grid_dx_meters' --obs_file $obs_dir/$pr_dir/profiles_${pgs}m_${region}_${saryears}_observed_flux_v${ver}.nc 600m_${CLIMATE}_${TYPE}/$pr_dir/profiles_${pgs}m_${region}_g600m_${CLIMATE}_${TYPE}_e_1.25_ppq_0.6_tefo_0.02_ssa_n_3.25_philow_5.0_hydro_null_100a.nc 1500m_${CLIMATE}_${TYPE}/${pr_dir}/profiles_${pgs}m_${region}_g1500m_${CLIMATE}_${TYPE}_e_1.25_ppq_0.6_tefo_0.02_ssa_n_3.25_philow_5.0_hydro_null_100a.nc 3600m_${CLIMATE}_${TYPE}/${pr_dir}/profiles_${pgs}m_${region}_g3600m_${CLIMATE}_${TYPE}_e_1.25_ppq_0.6_tefo_0.02_ssa_n_3.25_philow_5.0_hydro_null_100a.nc 4500m_${CLIMATE}_${TYPE}/${pr_dir}/profiles_${pgs}m_${region}_g4500m_${CLIMATE}_${TYPE}_e_1.25_ppq_0.6_tefo_0.02_ssa_n_3.25_philow_5.0_hydro_null_100a.nc
            for file in *_${var}_profile.pdf; do
                bfile=$(basename $file .pdf)
                mv $file ${bfile}_${pgs}m_${region}_${saryears}_${out_suffix}_sel_red_no_leg.pdf
            done
            ./flux-gate-analysis.py --legend none --plot_title -p 72mm -v $var --colormap Purples Sequential 5 0 --label_params 'grid_dx_meters' --obs_file $obs_dir/$pr_dir/profiles_${pgs}m_${region}_${saryears}_observed_flux_v${ver}.nc 600m_${CLIMATE}_${TYPE}/$pr_dir/profiles_${pgs}m_${region}_g600m_${CLIMATE}_${TYPE}_e_1.25_ppq_0.6_tefo_0.02_ssa_n_3.25_philow_5.0_hydro_null_100a.nc 1500m_${CLIMATE}_${TYPE}/${pr_dir}/profiles_${pgs}m_${region}_g1500m_${CLIMATE}_${TYPE}_e_1.25_ppq_0.6_tefo_0.02_ssa_n_3.25_philow_5.0_hydro_null_100a.nc 3600m_${CLIMATE}_${TYPE}/${pr_dir}/profiles_${pgs}m_${region}_g3600m_${CLIMATE}_${TYPE}_e_1.25_ppq_0.6_tefo_0.02_ssa_n_3.25_philow_5.0_hydro_null_100a.nc 4500m_${CLIMATE}_${TYPE}/${pr_dir}/profiles_${pgs}m_${region}_g4500m_${CLIMATE}_${TYPE}_e_1.25_ppq_0.6_tefo_0.02_ssa_n_3.25_philow_5.0_hydro_null_100a.nc
            for file in *_${var}_profile.pdf; do
                bfile=$(basename $file .pdf)
                mv $file ${bfile}_${pgs}m_${region}_${saryears}_${out_suffix}_sel_purple_no_leg.pdf
            done
            # with legend
          ./flux-gate-analysis.py --legend regress --plot_title -p 72mm -v $var --colormap Greens Sequential 5 0 --do_regress --label_params 'grid_dx_meters' --obs_file $obs_dir/$pr_dir/profiles_${pgs}m_${region}_${saryears}_observed_flux_v${ver}.nc 600m_${CLIMATE}_${TYPE}/$pr_dir/profiles_${pgs}m_${region}_g600m_${CLIMATE}_${TYPE}_e_1.25_ppq_0.6_tefo_0.02_ssa_n_3.25_philow_5.0_hydro_null_100a.nc 1500m_${CLIMATE}_${TYPE}/${pr_dir}/profiles_${pgs}m_${region}_g1500m_${CLIMATE}_${TYPE}_e_1.25_ppq_0.6_tefo_0.02_ssa_n_3.25_philow_5.0_hydro_null_100a.nc 3600m_${CLIMATE}_${TYPE}/${pr_dir}/profiles_${pgs}m_${region}_g3600m_${CLIMATE}_${TYPE}_e_1.25_ppq_0.6_tefo_0.02_ssa_n_3.25_philow_5.0_hydro_null_100a.nc 4500m_${CLIMATE}_${TYPE}/${pr_dir}/profiles_${pgs}m_${region}_g4500m_${CLIMATE}_${TYPE}_e_1.25_ppq_0.6_tefo_0.02_ssa_n_3.25_philow_5.0_hydro_null_100a.nc
            for file in *_${var}_profile.pdf; do
                bfile=$(basename $file .pdf)
                mv $file ${bfile}_${pgs}m_${region}_${saryears}_${out_suffix}_sel_green.pdf
            done
            ./flux-gate-analysis.py --legend regress --plot_title -p 72mm -v $var --colormap Blues Sequential 5 0 --do_regress --label_params 'grid_dx_meters' --obs_file $obs_dir/$pr_dir/profiles_${pgs}m_${region}_${saryears}_observed_flux_v${ver}.nc 600m_${CLIMATE}_${TYPE}/$pr_dir/profiles_${pgs}m_${region}_g600m_${CLIMATE}_${TYPE}_e_1.25_ppq_0.6_tefo_0.02_ssa_n_3.25_philow_5.0_hydro_null_100a.nc 1500m_${CLIMATE}_${TYPE}/${pr_dir}/profiles_${pgs}m_${region}_g1500m_${CLIMATE}_${TYPE}_e_1.25_ppq_0.6_tefo_0.02_ssa_n_3.25_philow_5.0_hydro_null_100a.nc 3600m_${CLIMATE}_${TYPE}/${pr_dir}/profiles_${pgs}m_${region}_g3600m_${CLIMATE}_${TYPE}_e_1.25_ppq_0.6_tefo_0.02_ssa_n_3.25_philow_5.0_hydro_null_100a.nc 4500m_${CLIMATE}_${TYPE}/${pr_dir}/profiles_${pgs}m_${region}_g4500m_${CLIMATE}_${TYPE}_e_1.25_ppq_0.6_tefo_0.02_ssa_n_3.25_philow_5.0_hydro_null_100a.nc
            for file in *_${var}_profile.pdf; do
                bfile=$(basename $file .pdf)
                mv $file ${bfile}_${pgs}m_${region}_${saryears}_${out_suffix}_sel_blue.pdf
            done
            ./flux-gate-analysis.py --legend regress --plot_title -p 72mm -v $var --colormap Reds Sequential 5 0 --do_regress --label_params 'grid_dx_meters' --obs_file $obs_dir/$pr_dir/profiles_${pgs}m_${region}_${saryears}_observed_flux_v${ver}.nc 600m_${CLIMATE}_${TYPE}/$pr_dir/profiles_${pgs}m_${region}_g600m_${CLIMATE}_${TYPE}_e_1.25_ppq_0.6_tefo_0.02_ssa_n_3.25_philow_5.0_hydro_null_100a.nc 1500m_${CLIMATE}_${TYPE}/${pr_dir}/profiles_${pgs}m_${region}_g1500m_${CLIMATE}_${TYPE}_e_1.25_ppq_0.6_tefo_0.02_ssa_n_3.25_philow_5.0_hydro_null_100a.nc 3600m_${CLIMATE}_${TYPE}/${pr_dir}/profiles_${pgs}m_${region}_g3600m_${CLIMATE}_${TYPE}_e_1.25_ppq_0.6_tefo_0.02_ssa_n_3.25_philow_5.0_hydro_null_100a.nc 4500m_${CLIMATE}_${TYPE}/${pr_dir}/profiles_${pgs}m_${region}_g4500m_${CLIMATE}_${TYPE}_e_1.25_ppq_0.6_tefo_0.02_ssa_n_3.25_philow_5.0_hydro_null_100a.nc
            for file in *_${var}_profile.pdf; do
                bfile=$(basename $file .pdf)
                mv $file ${bfile}_${pgs}m_${region}_${saryears}_${out_suffix}_sel_red.pdf
            done
            ./flux-gate-analysis.py --legend regress --plot_title -p 72mm -v $var --colormap Purples Sequential 5 0 --label_params 'grid_dx_meters' --obs_file $obs_dir/$pr_dir/profiles_${pgs}m_${region}_${saryears}_observed_flux_v${ver}.nc 600m_${CLIMATE}_${TYPE}/$pr_dir/profiles_${pgs}m_${region}_g600m_${CLIMATE}_${TYPE}_e_1.25_ppq_0.6_tefo_0.02_ssa_n_3.25_philow_5.0_hydro_null_100a.nc 1500m_${CLIMATE}_${TYPE}/${pr_dir}/profiles_${pgs}m_${region}_g1500m_${CLIMATE}_${TYPE}_e_1.25_ppq_0.6_tefo_0.02_ssa_n_3.25_philow_5.0_hydro_null_100a.nc 3600m_${CLIMATE}_${TYPE}/${pr_dir}/profiles_${pgs}m_${region}_g3600m_${CLIMATE}_${TYPE}_e_1.25_ppq_0.6_tefo_0.02_ssa_n_3.25_philow_5.0_hydro_null_100a.nc 4500m_${CLIMATE}_${TYPE}/${pr_dir}/profiles_${pgs}m_${region}_g4500m_${CLIMATE}_${TYPE}_e_1.25_ppq_0.6_tefo_0.02_ssa_n_3.25_philow_5.0_hydro_null_100a.nc
            for file in *_${var}_profile.pdf; do
                bfile=$(basename $file .pdf)
                mv $file ${bfile}_${pgs}m_${region}_${saryears}_${out_suffix}_sel_purple.pdf
            done
            for file in pearson_r_experiment_*_${var}.pdf; do
                bfile=$(basename $file .pdf)
                mv $file ${bfile}_${pgs}m_${region}_${saryears}_${out_suffix}.pdf
            done
            ./flux-gate-analysis.py --do_regress --legend short --plot_title -p 72mm -v $var --colormap Purples Sequential 7 0 --label_params 'grid_dx_meters' --obs_file $obs_dir/$pr_dir/profiles_${pgs}m_${region}_${saryears}_observed_flux_v${ver}.nc 600m_${CLIMATE}_${TYPE}/$pr_dir/profiles_${pgs}m_${region}_g600m_${CLIMATE}_${TYPE}_e_1.25_ppq_0.6_tefo_0.02_ssa_n_3.25_philow_5.0_hydro_null_100a.nc 900m_${CLIMATE}_${TYPE}/$pr_dir/profiles_${pgs}m_${region}_g900m_${CLIMATE}_${TYPE}_e_1.25_ppq_0.6_tefo_0.02_ssa_n_3.25_philow_5.0_hydro_null_100a.nc 1200m_${CLIMATE}_${TYPE}/${pr_dir}/profiles_${pgs}m_${region}_g1200m_${CLIMATE}_${TYPE}_e_1.25_ppq_0.6_tefo_0.02_ssa_n_3.25_philow_5.0_hydro_null_100a.nc 1500m_${CLIMATE}_${TYPE}/${pr_dir}/profiles_${pgs}m_${region}_g1500m_${CLIMATE}_${TYPE}_e_1.25_ppq_0.6_tefo_0.02_ssa_n_3.25_philow_5.0_hydro_null_100a.nc 1800m_${CLIMATE}_${TYPE}/${pr_dir}/profiles_${pgs}m_${region}_g1800m_${CLIMATE}_${TYPE}_e_1.25_ppq_0.6_tefo_0.02_ssa_n_3.25_philow_5.0_hydro_null_100a.nc 3600m_${CLIMATE}_${TYPE}/${pr_dir}/profiles_${pgs}m_${region}_g3600m_${CLIMATE}_${TYPE}_e_1.25_ppq_0.6_tefo_0.02_ssa_n_3.25_philow_5.0_hydro_null_100a.nc 4500m_${CLIMATE}_${TYPE}/${pr_dir}/profiles_${pgs}m_${region}_g4500m_${CLIMATE}_${TYPE}_e_1.25_ppq_0.6_tefo_0.02_ssa_n_3.25_philow_5.0_hydro_null_100a.nc
            for file in *_${var}_profile.pdf; do
                bfile=$(basename $file .pdf)
                mv $file ${bfile}_${pgs}m_${region}_${saryears}_${out_suffix}_purple.pdf
            done
            for file in pearson_r_experiment_*_${var}.pdf; do
                bfile=$(basename $file .pdf)
                mv $file ${bfile}_${pgs}m_${region}_${saryears}_${out_suffix}.pdf
            done
            mv rmsd_cum_table.tex rmsd_cum_table_${var}_${pgs}m_${out_suffix}.tex
            mv rmsd_regression.pdf rmsd_regression_${var}_${pgs}m_${out_suffix}.pdf
            mv r2_regression.pdf r2_regression_${var}_${pgs}m_${out_suffix}.pdf
            mv pearson_r_regression.pdf pearson_r_regression_${var}_${pgs}m_${out_suffix}.pdf

            mv statistics.shp statistics_${var}_${pgs}m_${region}_${saryears}_${out_suffix}.shp
            mv statistics.dbf statistics_${var}_${pgs}m_${region}_${saryears}_${out_suffix}.dbf
            mv statistics.prj statistics_${var}_${pgs}m_${region}_${saryears}_${out_suffix}.prj
            mv statistics.shx statistics_${var}_${pgs}m_${region}_${saryears}_${out_suffix}.shx
        done
        # for var in  "flux_normal"; do
        #     ./flux-gate-analysis.py --legend long --plot_title -p 72mm -v $var --colormap YlGnBu Sequential 7 0 --do_regress --label_params 'grid_dx_meters' --obs_file $obs_dir/$pr_dir/profiles_${pgs}m_${region}_${saryears}_observed_flux_v${ver}.nc 600m_${CLIMATE}_${TYPE}/$pr_dir/profiles_${pgs}m_${region}_g600m_${CLIMATE}_${TYPE}_e_1.25_ppq_0.6_tefo_0.02_ssa_n_3.25_philow_5.0_hydro_null_100a.nc 900m_${CLIMATE}_${TYPE}/$pr_dir/profiles_${pgs}m_${region}_g900m_${CLIMATE}_${TYPE}_e_1.25_ppq_0.6_tefo_0.02_ssa_n_3.25_philow_5.0_hydro_null_100a.nc 1200m_${CLIMATE}_${TYPE}/${pr_dir}/profiles_${pgs}m_${region}_g1200m_${CLIMATE}_${TYPE}_e_1.25_ppq_0.6_tefo_0.02_ssa_n_3.25_philow_5.0_hydro_null_100a.nc 1500m_${CLIMATE}_${TYPE}/${pr_dir}/profiles_${pgs}m_${region}_g1500m_${CLIMATE}_${TYPE}_e_1.25_ppq_0.6_tefo_0.02_ssa_n_3.25_philow_5.0_hydro_null_100a.nc 1800m_${CLIMATE}_${TYPE}/${pr_dir}/profiles_${pgs}m_${region}_g1800m_${CLIMATE}_${TYPE}_e_1.25_ppq_0.6_tefo_0.02_ssa_n_3.25_philow_5.0_hydro_null_100a.nc 3600m_${CLIMATE}_${TYPE}/${pr_dir}/profiles_${pgs}m_${region}_g3600m_${CLIMATE}_${TYPE}_e_1.25_ppq_0.6_tefo_0.02_ssa_n_3.25_philow_5.0_hydro_null_100a.nc 4500m_${CLIMATE}_${TYPE}/${pr_dir}/profiles_${pgs}m_${region}_g4500m_${CLIMATE}_${TYPE}_e_1.25_ppq_0.6_tefo_0.02_ssa_n_3.25_philow_5.0_hydro_null_100a.nc
        #     for file in *_${var}_profile.pdf; do
        #         bfile=$(basename $file .pdf)
        #         mv $file ${bfile}_${pgs}m_${region}_${saryears}_${out_suffix}.pdf
        #     done
        done
    done
done
