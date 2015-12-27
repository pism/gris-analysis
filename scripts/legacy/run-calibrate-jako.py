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

ver=2_1985
CLIMATE=const
saryears=1985

GRID=1200
tl_dir=relax_${GRID}m_${CLIMATE}

for region in "jakobshavn"; do
    for var in "land_ice_thickness" "velsurf_normal"; do
        mytype=ctrl
        out_suffix=grid_res_${saryears}
        # ~/base/pypismtools/scripts/flowline-plot.py -p twocol  --label_params ocean_forcing_type,fracture_density_softening_lower_limit,ssa_enhancement_factor -o ${mytype}_lm -v ${var} --obs_file $obs_dir/$pr_dir/profiles_${pgs}m_${region}_${saryears}_observed_flux_v${ver}.nc $tl_dir/$pr_dir/profiles_${pgs}m_${region}_g${GRID}m_${CLIMATE}_v${ver}_${mytype}_20a_ssa_e_*fsoft*lm_*.nc
         ~/base/pypismtools/scripts/flowline-plot.py -p twocol  --label_params grid_dx_meters,fracture_density_softening_lower_limit,ssa_enhancement_factor -o ${out_suffix} -v ${var} --obs_file $obs_dir/$pr_dir/profiles_${pgs}m_${region}_${saryears}_observed_flux_v${ver}.nc relax_900m_const/profiles/profiles_250m_jakobshavn_g900m_const_v2_1985_ctrl_20a_ssa_e_1.0_k_1e18_fsoft_0.95_calving_eigen_calving_100_ocean_const_m20.nc relax_1200m_const/profiles/profiles_250m_jakobshavn_g1200m_const_v2_1985_ctrl_20a_ssa_e_1.0_k_1e18_fsoft_0.95_calving_eigen_calving_100_ocean_constrsb_m20.nc relax_900m_const/profiles/profiles_250m_jakobshavn_g900m_const_v2_1985_ctrl_20a_ssa_e_1.0_k_1e18_fsoft_1.0_calving_eigen_calving_100_ocean_const_m20.nc
        # ~/base/fastflow-paper/analysis/flux-gate-analysis.py -p twocol  --legend exp --label_params grid_dx_meters,fracture_density_softening_lower_limit,ssa_enhancement_factor -v ${var} --obs_file $obs_dir/$pr_dir/profiles_${pgs}m_${region}_${saryears}_observed_flux_v${ver}.nc relax_900m_const/profiles/profiles_250m_jakobshavn_g900m_const_v2_1985_ctrl_20a_ssa_e_1.0_k_1e18_fsoft_0.95_calving_eigen_calving_100_ocean_const_m20.nc relax_1200m_const/profiles/profiles_250m_jakobshavn_g1200m_const_v2_1985_ctrl_20a_ssa_e_1.0_k_1e18_fsoft_0.95_calving_eigen_calving_100_ocean_constrsb_m20.nc
        # for file in C?_${var}_profile.pdf; do            
        #     bfile=$(basename $file .pdf)
        #     mv $file ${bfile}_${pgs}m_${region}_${saryears}_${out_suffix}.pdf
        # done
    done
done

ver=2
CLIMATE=const
saryears=2008-2009

GRID=1200
tl_dir=relax_${GRID}m_${CLIMATE}

for region in "jakobshavn"; do
    for var in "land_ice_thickness" "velsurf_normal"; do
        mytype=ctrl
        out_suffix=grid_res_${saryears}
        ~/base/pypismtools/scripts/flowline-plot.py -o $out_suffix -p twocol  --legend default --label_params grid_dx_meters,fracture_density_softening_lower_limit,ssa_enhancement_factor -v ${var} --obs_file $obs_dir/$pr_dir/profiles_${pgs}m_${region}_${saryears}_observed_flux_v${ver}.nc  900m_const_ctrl_v2/profiles/profiles_250m_jakobshavn_g900m_const_ctrl_v2_sia_e_1.25_ppq_0.6_tefo_0.02_ssa_n_3.25_ssa_e_1.0_philow_5.0_hydro_null_100a.nc  1200m_const_ctrl_v2/profiles/profiles_250m_jakobshavn_g1200m_const_ctrl_v2_sia_e_1.25_ppq_0.6_tefo_0.02_ssa_n_3.25_ssa_e_1.0_philow_5.0_hydro_null_100a.nc 
        # ~/base/fastflow-paper/analysis/flux-gate-analysis.py -p twocol  --legend exp --label_params grid_dx_meters,fracture_density_softening_lower_limit,ssa_enhancement_factor -v ${var} --obs_file $obs_dir/$pr_dir/profiles_${pgs}m_${region}_${saryears}_observed_flux_v${ver}.nc  900m_const_ctrl_v2/profiles/profiles_250m_jakobshavn_g900m_const_ctrl_v2_sia_e_1.25_ppq_0.6_tefo_0.02_ssa_n_3.25_ssa_e_1.0_philow_5.0_hydro_null_100a.nc  1200m_const_ctrl_v2/profiles/profiles_250m_jakobshavn_g1200m_const_ctrl_v2_sia_e_1.25_ppq_0.6_tefo_0.02_ssa_n_3.25_ssa_e_1.0_philow_5.0_hydro_null_100a.nc
    done
done


exit

for region in "jakobshavn"; do
    for var in "land_ice_thickness" "velsurf_mag"; do
        mytype=ctrl
        # ~/base/pypismtools/scripts/flowline-plot.py -p twocol  --label_params ocean_forcing_type,fracture_density_softening_lower_limit,ssa_enhancement_factor -o ${mytype}_lm -v ${var} --obs_file $obs_dir/$pr_dir/profiles_${pgs}m_${region}_${saryears}_observed_flux_v${ver}.nc $tl_dir/$pr_dir/profiles_${pgs}m_${region}_g${GRID}m_${CLIMATE}_v${ver}_${mytype}_20a_ssa_e_*fsoft*lm_*.nc
        # ~/base/pypismtools/scripts/flowline-plot.py -p twocol  --label_params grid_dx_meters,fracture_density_softening_lower_limit,ssa_enhancement_factor -o ${mytype}_fsoft -v ${var} --obs_file $obs_dir/$pr_dir/profiles_${pgs}m_${region}_${saryears}_observed_flux_v${ver}.nc $tl_dir/$pr_dir/profiles_${pgs}m_${region}_g${GRID}m_${CLIMATE}_v${ver}_${mytype}_20a_ssa_e_*fsoft*constrsb_m20.nc
        ~/base/fastflow-paper/analysis/flux-gate-analysis.py -p twocol  --label_params grid_dx_meters,fracture_density_softening_lower_limit,ssa_enhancement_factor -v ${var} --obs_file $obs_dir/$pr_dir/profiles_${pgs}m_${region}_${saryears}_observed_flux_v${ver}.nc relax_900m_const/profiles/profiles_250m_jakobshavn_g900m_const_v2_1985_ctrl_20a_ssa_e_1.0_k_1e18_fsoft_0.95_calving_eigen_calving_100_ocean_const_m20.nc relax_1200m_const/profiles/profiles_250m_jakobshavn_g1200m_const_v2_1985_ctrl_20a_ssa_e_1.0_k_1e18_fsoft_0.95_calving_eigen_calving_100_ocean_constrsb_m20.nc
    done
done

for region in "jakobshavn-flowline"; do
    for var in "land_ice_thickness" "velsurf_mag"; do
        mytype=ctrl
        # ~/base/pypismtools/scripts/flowline-plot.py -p twocol  --label_params ocean_forcing_type,fracture_density_softening_lower_limit,ssa_enhancement_factor -o ${mytype}_lm -v ${var} --obs_file $obs_dir/$pr_dir/profiles_${pgs}m_${region}_${saryears}_observed_flux_v${ver}.nc $tl_dir/$pr_dir/profiles_${pgs}m_${region}_g${GRID}m_${CLIMATE}_v${ver}_${mytype}_20a_ssa_e_*fsoft*lm_*.nc
        ~/base/pypismtools/scripts/flowline-plot.py -p twocol  --label_params grid_dx_meters,fracture_density_softening_lower_limit,ssa_enhancement_factor -o ${mytype}_fsoft -v ${var} --obs_file $obs_dir/$pr_dir/profiles_${pgs}m_${region}_${saryears}_observed_flux_v${ver}.nc relax_900m_const/profiles/profiles_250m_jakobshavn-flowline_g900m_const_v2_1985_ctrl_20a_ssa_e_1.0_k_1e18_fsoft_0.95_calving_eigen_calving_100_ocean_const_m20.nc relax_1200m_const/profiles/profiles_250m_jakobshavn-flowline_g1200m_const_v2_1985_ctrl_20a_ssa_e_1.0_k_1e18_fsoft_0.95_calving_eigen_calving_100_ocean_constrsb_m20.nc
        # ~/base/pypismtools/scripts/flowline-plot.py -p twocol  --label_params grid_dx_meters,fracture_density_softening_lower_limit,ssa_enhancement_factor -o ${mytype}_fsoft -v ${var} --obs_file $obs_dir/$pr_dir/profiles_${pgs}m_${region}_${saryears}_observed_flux_v${ver}.nc relax_900m_const/profiles/profiles_250m_jakobshavn-flowline_g900m_const_v2_1985_ctrl_20a_ssa_e_1.0_k_1e18_fsoft_0.95_calving_eigen_calving_100_ocean_const_m20.nc relax_1200m_const/profiles/profiles_250m_jakobshavn-flowline_g1200m_const_v2_1985_ctrl_20a_ssa_e_1.0_k_1e18_fsoft_0.95_calving_eigen_calving_100_ocean_constrsb_m20.nc
    done
done


exit

ver=2
CLIMATE=const
saryears=2008-2009
TYPE=ctrl_v${ver}
GRID=1200
tl_dir=${GRID}m_${CLIMATE}_${TYPE}

for region in "jakobshavn-main-flowline"; do
    for var in "land_ice_thickness" "velsurf_mag"; do
        mytype=ctrl_$saryears
        ~/base/pypismtools/scripts/flowline-plot.py -p twocol  --label_params ssa_enhancement_factor,grid_dx_meters -o ${mytype} -v ${var} --obs_file $obs_dir/$pr_dir/profiles_${pgs}m_${region}_${saryears}_observed_flux_v${ver}.nc 900m_const_ctrl_v2/profiles/profiles_250m_jakobshavn-main-flowline_g900m_const_ctrl_v2_sia_e_1.25_ppq_0.6_tefo_0.02_ssa_n_3.25_ssa_e_1.0_philow_5.0_hydro_null_100a.nc $tl_dir/$pr_dir/profiles_${pgs}m_${region}_g${GRID}m_${CLIMATE}_${TYPE}_sia_e_1.25_ppq_0.6_tefo_0.02_ssa_n_3.25_ssa_e_1.0_philow_5.0_hydro_null_100a.nc
    done
done
