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

region="greenland"
# set region from argument 2
if [ -n "$2" ]; then
    region=$2
fi

ver=2
# set resolution from argument 3
if [ -n "$3" ]; then
    ver=$3
fi

pgs=250
# set resolution from argument 4
if [ -n "$4" ]; then
    pgs=$4
fi

CLIMATE=const
TYPE=ctrl   
ver=2

tl_dir=${GRID}m_${CLIMATE}_${TYPE}
nc_dir=processed
pr_dir=profiles
reg_dir=regional
obs_dir=observed  
fig_dir=figures
    
if [ ! -d ${tl_dir}/${pr_dir} ]; then
    mkdir -p ${tl_dir}/${pr_dir}
fi

if [ ! -d ${tl_dir}/${reg_dir} ]; then
    mkdir -p ${tl_dir}/${reg_dir}
fi

if [ ! -d ${obs_dir}/${pr_dir} ]; then
    mkdir -p ${obs_dir}/${pr_dir}
fi
source extract-observed-profiles.sh


for GRID in 900 1200 1500; do
    CLIMATE=const
    TYPE=ctrl_v${ver}
    
    tl_dir=${GRID}m_${CLIMATE}_${TYPE}
    nc_dir=processed
    pr_dir=profiles
    reg_dir=regional
    obs_dir=observed  
    fig_dir=figures
    
    if [ ! -d ${tl_dir}/${pr_dir} ]; then
        mkdir -p ${tl_dir}/${pr_dir}
    fi
    
    if [ ! -d ${tl_dir}/${reg_dir} ]; then
        mkdir -p ${tl_dir}/${reg_dir}
    fi
    
    source extract-profiles.sh
done


for GRID in 900; do
    CLIMATE=const
    TYPE=cresis_v${ver}
    
    tl_dir=${GRID}m_${CLIMATE}_${TYPE}
    nc_dir=processed
    pr_dir=profiles
    reg_dir=regional
    obs_dir=observed  
    fig_dir=figures
    
    if [ ! -d ${tl_dir}/${pr_dir} ]; then
        mkdir -p ${tl_dir}/${pr_dir}
    fi
    
    if [ ! -d ${tl_dir}/${reg_dir} ]; then
        mkdir -p ${tl_dir}/${reg_dir}
    fi
    
    source extract-profiles.sh
done
