#!/bin/bash

# run in green-hydro/run
# extract profiles for all experiments
set -e -x

# set overwrite from argument 1
# choose "1" and existing profiles will be re-computed, otherwise, only new files
# will be processed
if [ -n "$1" ]; then
    overwrite=$1
fi

pgs=250
# set resolution from argument 1
if [ -n "$2" ]; then
    pgs=$2
fi

CLIMATE=const

nc_dir=processed
pr_dir=profiles
reg_dir=regional
obs_dir=observed  
fig_dir=figures
for region in "jakobshavn" "jakobshavn-flowline"; do
        
    if [ ! -d ${obs_dir}/${pr_dir} ]; then
        mkdir -p ${obs_dir}/${pr_dir}
    fi
    ver=2_1985
    source extract-observed-1985-profiles.sh
done

for GRID in 900 1200 ; do
    CLIMATE=const
    
    tl_dir=relax_${GRID}m_${CLIMATE}
    nc_dir=processed
    pr_dir=profiles
    obs_dir=observed  
    for region in "jakobshavn" "jakobshavn-flowline"; do
    
        if [ ! -d ${tl_dir}/${pr_dir} ]; then
            mkdir -p ${tl_dir}/${pr_dir}
        fi
        
        if [ ! -d ${tl_dir}/${reg_dir} ]; then
            mkdir -p ${tl_dir}/${reg_dir}
        fi
        
        if [ ! -d ${obs_dir}/${pr_dir} ]; then
            mkdir -p ${obs_dir}/${pr_dir}
        fi
        source extract-profiles.sh
    done
done


