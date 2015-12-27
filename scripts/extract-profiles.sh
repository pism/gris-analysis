# Extract profiles from simulations

cd ${tl_dir}/${nc_dir}
for file in g${GRID}m_*; do
    if [ "${overwrite}" == "1" ]; then
        extract_profiles.py /Volumes/Isunnguata\ Sermia/data/data_sets/GreenlandFluxGates/${region}-flux-gates-${pgs}m.shp $file ../../$tl_dir/$pr_dir/profiles_${pgs}m_${region}_$file
        ncap2 -O -s "flux_normal=uflux*nx+vflux*ny; velsurf_normal=uvelsurf*nx+vvelsurf*ny" ../../$tl_dir/$pr_dir/profiles_${pgs}m_${region}_$file  ../../$tl_dir/$pr_dir/profiles_${pgs}m_${region}_$file
        ncatted -a long_name,flux_normal,o,c,"flux normal to gate" -a standard_name,flux_normal,d,, -a long_name,velsurf_normal,o,c,"surface speed normal to gate" ../../$tl_dir/$pr_dir/profiles_${pgs}m_${region}_$file
    else
        if [ -f "../../$tl_dir/$pr_dir/profiles_${pgs}m_${region}_$file" ]; then
            echo "$file exists, skipping"
        else
            extract_profiles.py /Volumes/Isunnguata\ Sermia/data/data_sets/GreenlandFluxGates/${region}-flux-gates-${pgs}m.shp $file ../../$tl_dir/$pr_dir/profiles_${pgs}m_${region}_$file
            ncap2 -O -s "flux_normal=uflux*nx+vflux*ny; velsurf_normal=uvelsurf*nx+vvelsurf*ny" ../../$tl_dir/$pr_dir/profiles_${pgs}m_${region}_$file  ../../$tl_dir/$pr_dir/profiles_${pgs}m_${region}_$file
            ncatted -a long_name,flux_normal,o,c,"flux normal to gate" -a standard_name,flux_normal,d,, -a long_name,velsurf_normal,o,c,"surface speed normal to gate" ../../$tl_dir/$pr_dir/profiles_${pgs}m_${region}_$file
        fi
    fi
done
cd ../../
