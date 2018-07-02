#!/bin/bash

set -x

for file in *.gz; do
    gunzip $file;
done

cdo -O  fldmean -yearmean -selseason,JJA -selvar,tas DMI-HIRHAM5_GL2_ERAI_1980_2014_TAS_DM.nc fldmean_tas_ym_jja_DMI-HIRHAM5_GL2_ERAI_1980_2014_TAS_DM.nc
cdo -O fldmean -yearsum -selvar,mrros DMI-HIRHAM5_GL2_ERAI_1980_2014_MRROS_DM.nc fldmean_ru_ys_jja_DMI-HIRHAM5_GL2_ERAI_1980_2014_MRROS_DM.nc
cdo -O merge fldmean_tas_ym_jja_DMI-HIRHAM5_GL2_ERAI_1980_2014_TAS_DM.nc  fldmean_ru_ys_jja_DMI-HIRHAM5_GL2_ERAI_1980_2014_MRROS_DM.nc  fldmean_ys_jja_run_ERAI.nc

for rcp in 45 85; do
    cdo -O mergetime GR6*2_ECEARTH_RCP${rcp}_20*_MRROS_DM.nc GR6c2_ECEARTH_RCP${rcp}_MRROS_DM.nc
    cdo -O mergetime GR6*2_ECEARTH_RCP${rcp}_20*_TAS_DM.nc GR6c2_ECEARTH_RCP${rcp}_TAS_DM.nc
    cdo -O fldmean -yearmean -selseason,JJA -selvar,tas GR6c2_ECEARTH_RCP${rcp}_TAS_DM.nc fldmean_tas_ym_jja_GR6c2_ECEARTH_RCP${rcp}_TAS_DM.nc
    cdo -O fldmean -yearsum -selvar,mrros  GR6c2_ECEARTH_RCP${rcp}_MRROS_DM.nc fldmean_mrros_ys_jja_GR6c2_ECEARTH_RCP${rcp}_MRROS_DM.nc
    cdo -O merge fldmean_tas_ym_jja_GR6c2_ECEARTH_RCP${rcp}_TAS_DM.nc fldmean_mrros_ys_jja_GR6c2_ECEARTH_RCP${rcp}_MRROS_DM.nc fldmean_ys_jja_run_RCP${rcp}.nc
done
