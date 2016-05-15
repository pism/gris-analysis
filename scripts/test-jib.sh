for run in ctrl cresisp; do
extract_profiles.py -a ~/base/gris-analysis/flux-gates/jib-sb-flux-gates-250m.shp 2016-05_hindcast_${run}/jakobshavn_g600m_pdd_lapse_v2_${run}_sia_e_3.0_ppq_0.6_tefo_0.02_calving_ocean_kill_2000-1-1_2008-1-1.nc profiles/profile_250m_jakobshavn_g600m_pdd_lapse_v2_${run}_sia_e_3.0_ppq_0.6_tefo_0.02_calving_ocean_kill_2000-1-1_2008-1-1.nc
ncap2 -O -s "velsurf_normal=u_ssa*nx+v_ssa*ny; velsurf_mag=sqrt(u_ssa^2+v_ssa^2);"  profiles/profile_250m_jakobshavn_g600m_pdd_lapse_v2_${run}_sia_e_3.0_ppq_0.6_tefo_0.02_calving_ocean_kill_2000-1-1_2008-1-1.nc  profiles/profile_250m_jakobshavn_g600m_pdd_lapse_v2_${run}_sia_e_3.0_ppq_0.6_tefo_0.02_calving_ocean_kill_2000-1-1_2008-1-1.nc
done
GRID=450
for run in cresisp; do
extract_profiles.py -a ~/base/gris-analysis/flux-gates/jib-sb-flux-gates-250m.shp 2016-05_hindcast_${run}2/jakobshavn_g450m_pdd_lapse_v2_${run}_sia_e_3.0_ppq_0.6_tefo_0.02_calving_ocean_kill_2000-1-1_2008-1-1.nc profiles/profile_250m_jakobshavn_g450m_pdd_lapse_v2_${run}_sia_e_3.0_ppq_0.6_tefo_0.02_calving_ocean_kill_2000-1-1_2008-1-1.nc
ncap2 -O -s "velsurf_normal=u_ssa*nx+v_ssa*ny; velsurf_mag=sqrt(u_ssa^2+v_ssa^2);"  profiles/profile_250m_jakobshavn_g450m_pdd_lapse_v2_${run}_sia_e_3.0_ppq_0.6_tefo_0.02_calving_ocean_kill_2000-1-1_2008-1-1.nc  profiles/profile_250m_jakobshavn_g450m_pdd_lapse_v2_${run}_sia_e_3.0_ppq_0.6_tefo_0.02_calving_ocean_kill_2000-1-1_2008-1-1.nc
done
extract_profiles.py -a ~/base/gris-analysis/flux-gates/jib-sb-flux-gates-250m.shp /Volumes/Isunnguata_Sermia/green-hydro/run/observed/processed/Greenland_150m_2008-2009_observed_flux_v2.nc profiles/profile_250m_jib_obs.nc
ncap2 -O -s "velsurf_normal=uvelsurf*nx+vvelsurf*ny;"  profiles/profile_250m_jib_obs.nc  profiles/profile_250m_jib_obs.nc
 
~/base/gris-analysis/scripts/flux-gate-analysis.py -v velsurf_mag --legend short --label_params grid_dx_meters --colormap Set1 qualitative 4 0 --obs_file profiles/profile_250m_jib_obs.nc profiles/profile_250m_jak*.nc
