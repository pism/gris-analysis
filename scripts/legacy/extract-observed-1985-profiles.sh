# Extract profiles from speed and thickness observations and calculate flux

ucigrid=150

# Extract SAR from Motyka
extract_profiles.py  --all_variables /Volumes/Isunnguata\ Sermia/data/data_sets/GreenlandFluxGates/${region}-flux-gates-${pgs}m.shp ${obs_dir}/${nc_dir}/jakobshavn_surface_velocities_1985.nc $obs_dir/$pr_dir/profiles_${pgs}m_${region}_velocities_1985.nc

# Extract ice thickness from original MCB
extract_profiles.py  --all_variables /Volumes/Isunnguata\ Sermia/data/data_sets/GreenlandFluxGates/${region}-flux-gates-${pgs}m.shp ${obs_dir}/${nc_dir}/Greenland_${ucigrid}m_mcb_jpl_v${ver}_hole.nc $obs_dir/$pr_dir/profiles_${pgs}m_${region}_${ucigrid}m_mcb_jpl_v${ver}.nc

# Extract ice thickness from boot file
# extract_profiles.py  --all_variables /Volumes/Isunnguata\ Sermia/data/data_sets/GreenlandFluxGates/${region}-flux-gates-${pgs}m.shp ${obs_dir}/${nc_dir}/pism_Greenland_${GRID}m_mcb_jpl_v${ver}_${TYPE}.nc $obs_dir/$pr_dir/profiles_${pgs}m_${region}_${GRID}m_mcb_jpl_v${ver}_${TYPE}.nc

# Calculate fluxes by multiplying SAR and MCB
# We use vertically-average speed = surface speed
ncks -O $obs_dir/$pr_dir/profiles_${pgs}m_${region}_velocities_1985.nc $obs_dir/$pr_dir/profiles_${pgs}m_${region}_1985_observed_flux_v${ver}.nc
ncks -A $obs_dir/$pr_dir/profiles_${pgs}m_${region}_${ucigrid}m_mcb_jpl_v${ver}.nc $obs_dir/$pr_dir/profiles_${pgs}m_${region}_1985_observed_flux_v${ver}.nc
ncatted -a _FillValue,errbed,d,,   $obs_dir/$pr_dir/profiles_${pgs}m_${region}_1985_observed_flux_v${ver}.nc
# set a minimum error in ice thickness of 10m.
# Motyka et al (2011) give an error estimate of 0.3 m/day = 110 m/year
ncap2 -O -s "where(errbed==-9999) errbed=10; velsurf_mag_error=0*velsurf_mag; velsurf_mag_error=110."  $obs_dir/$pr_dir/profiles_${pgs}m_${region}_1985_observed_flux_v${ver}.nc  $obs_dir/$pr_dir/profiles_${pgs}m_${region}_1985_observed_flux_v${ver}.nc
# we assume annually-average speeds are 0.98-1.02 winter speeds
ncap2 -O -s "flux_normal=(uvelsurf*nx+vvelsurf*ny)*thickness; velsurf_normal=(uvelsurf*nx+vvelsurf*ny); flux_mag = thickness*velsurf_mag; flux_mag_error=flux_mag*sqrt(((velsurf_mag_error+velsurf_mag*0.02)/(velsurf_mag+1e-9))^2+(errbed/(thickness+1e-9))^2); where(flux_mag_error<0) flux_mag_error=flux_mag*sqrt((20/(velsurf_mag+1e-9))^2+(errbed/(thickness+1e-9))^2); velsurf_normal_error=velsurf_normal*sqrt((velsurf_mag_error/velsurf_normal)^2+0.02^2); thickness_error=errbed; flux_normal_error=flux_normal*sqrt(((velsurf_normal_error+velsurf_mag*0.02)/(velsurf_normal+1e-9))^2+(errbed/(thickness+1e-9))^2); where(flux_normal_error<0) flux_mag_error=flux_normal*sqrt((20/(velsurf_normal+1e-9))^2+(errbed/(thickness+1e-9))^2);"  $obs_dir/$pr_dir/profiles_${pgs}m_${region}_1985_observed_flux_v${ver}.nc  $obs_dir/$pr_dir/profiles_${pgs}m_${region}_1985_observed_flux_v${ver}.nc
ncatted -a long_name,velsurf_normal_error,o,c,"error in speed" -a standard_name,'^flux_normal.?',d,, -a standard_name,'^flux_mag.?',d,, -a units,'^flux_normal.?',o,c,"m2 yr-1" -a units,'^flux_mag.?',o,c,"m2 yr-1" -a long_name,flux_mag,o,c,"magnitude of flux" -a long_name,flux_normal,o,c,"flux normal to gate" -a long_name,velsurf_normal,o,c,"speed normal to gate" -a long_name,flux_mag_error,o,c,"error in magnitude of flux" -a long_name,flux_normal_error,o,c,"error in flux normal to gate" -a long_name,velsurf_normal_error,o,c,"error in speed normal to gate" $obs_dir/$pr_dir/profiles_${pgs}m_${region}_1985_observed_flux_v${ver}.nc

