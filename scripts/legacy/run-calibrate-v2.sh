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
TYPE=ctrl_v${ver}

GRID=1500
tl_dir=${GRID}m_${CLIMATE}_${TYPE}

out_suffix=${GRID}m_calib_tefo
for region in "greenland"; do
    for saryears in "2008-2009"; do
        for var in "velsurf_normal"; do
            ./flux-gate-analysis.py --plot_title -p twocol --legend exp --label_params 'pseudo_plastic_q,till_effective_fraction_overburden'  -v ${var} --obs_file $obs_dir/$pr_dir/profiles_${pgs}m_${region}_${saryears}_observed_flux.nc  $tl_dir/$pr_dir/profiles_${pgs}m_${region}_g${GRID}m_${CLIMATE}_${TYPE}_sia_e_*tefo_0.025*.nc $tl_dir/$pr_dir/profiles_${pgs}m_${region}_g${GRID}m_${CLIMATE}_${TYPE}_sia_e_*tefo_0.03*.nc
            for file in *_${var}_profile.pdf; do
                bfile=$(basename $file .pdf)
                mv $file ${bfile}_${pgs}m_${region}_${saryears}_${out_suffix}.pdf
            done
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

exit

out_suffix=${GRID}m_calib_ppq_0.6
for region in "greenland"; do
    for saryears in "2008-2009"; do
        for var in "velsurf_normal"; do
            ./flux-gate-analysis.py --plot_title -p twocol --legend short --label_params 'ssa_Glen_exponent,pseudo_plastic_q,till_effective_fraction_overburden,sia_enhancement_factor,ssa_enhancement_factor'  -v ${var} --obs_file $obs_dir/$pr_dir/profiles_${pgs}m_${region}_${saryears}_observed_flux.nc  $tl_dir/$pr_dir/profiles_${pgs}m_${region}_g${GRID}m_${CLIMATE}_${TYPE}_sia_e_*ppq_0.6*.nc
            for file in *_${var}_profile.pdf; do
                bfile=$(basename $file .pdf)
                mv $file ${bfile}_${pgs}m_${region}_${saryears}_${out_suffix}.pdf
            done
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

out_suffix=${GRID}m_calib_ppq_0.5
for region in "greenland"; do
    for saryears in "2008-2009"; do
        for var in "velsurf_normal"; do
            ./flux-gate-analysis.py --plot_title -p twocol --legend short --label_params 'ssa_Glen_exponent,pseudo_plastic_q,till_effective_fraction_overburden,sia_enhancement_factor,ssa_enhancement_factor'  -v ${var} --obs_file $obs_dir/$pr_dir/profiles_${pgs}m_${region}_${saryears}_observed_flux.nc  $tl_dir/$pr_dir/profiles_${pgs}m_${region}_g${GRID}m_${CLIMATE}_${TYPE}_sia_e_*ppq_0.5*.nc
            for file in *_${var}_profile.pdf; do
                bfile=$(basename $file .pdf)
                mv $file ${bfile}_${pgs}m_${region}_${saryears}_${out_suffix}.pdf
            done
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

exit

out_suffix=${GRID}m_calib
for region in "greenland"; do
    for saryears in "2008-2009"; do
        for var in "velsurf_normal"; do
            ./flux-gate-analysis.py --plot_title -p 72mm --legend short --no_figures --export_table_file ${var}-gate-table-all-${pgs}m.tex --label_params 'ssa_Glen_exponent,pseudo_plastic_q,till_effective_fraction_overburden,sia_enhancement_factor,ssa_enhancement_factor'  -v ${var} --obs_file $obs_dir/$pr_dir/profiles_${pgs}m_${region}_${saryears}_observed_flux.nc  $tl_dir/$pr_dir/profiles_${pgs}m_${region}_g${GRID}m_${CLIMATE}_${TYPE}_sia_e_*.nc
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

exit

for region in "greenland"; do
    for saryears in "2008-2009"; do
        for var in "velsurf_normal"; do
            ./flux-gate-analysis.py --plot_title -p 72mm --legend short  -no_figures --export_table_file ${var}-gate-table-all-${pgs}m.tex --label_params 'ssa_Glen_exponent,pseudo_plastic_q'  -v ${var} --obs_file $obs_dir/$pr_dir/profiles_${pgs}m_${region}_${saryears}_observed_flux.nc $tl_dir/$pr_dir/profiles_${pgs}m_${region}_g${GRID}m_${CLIMATE}_${TYPE}_e_1.25_ppq_*_tefo_0.02_ssa_n_*_*_hydro_null_100a.nc
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
