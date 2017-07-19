#!/bin/bash

set -x 
for gcm in CanESM2 MIROC5 NorESM1; do
    for rcp in 45 85; do
        mdir=${gcm}-rcp${rcp}_2006-2100_25km/monthly_raw_outputs_at_25km
        cdo -O mergetime ${mdir}/MARv3.5.2_${gcm}-rcp${rcp}_2*.nc MARv3.5.2_${gcm}-rcp${rcp}_2006-2100.nc
        cdo -O fldmean -yearmean -selseason,JJA -selvar,ST MARv3.5.2_${gcm}-rcp${rcp}_2006-2100.nc fldmean_st_ym_jja_MARv3.5.2_${gcm}-rcp${rcp}_2006-2100.nc
        cdo -O fldmean -yearsum -selvar,RU MARv3.5.2_${gcm}-rcp${rcp}_2006-2100.nc fldmean_ru_ys_jja_MARv3.5.2_${gcm}-rcp${rcp}_2006-2100.nc
        cdo -O merge fldmean_st_ym_jja_MARv3.5.2_${gcm}-rcp${rcp}_2006-2100.nc fldmean_ru_ys_jja_MARv3.5.2_${gcm}-rcp${rcp}_2006-2100.nc ru_st_MARv3.5.2_${gcm}-rcp${rcp}_2006-2100.nc
        if [ $gcm = 'CanESM2' ] && [ $rcp = '85' ]; then
            ncatted -a forcing,global,o,c,"${gcm}-rcp${rcp}" ru_st_MARv3.5.2_${gcm}-rcp${rcp}_2006-2100.nc
        fi
        if [ $gcm = 'NorESM1' ] && [ $rcp = '45' ]; then
            ncatted -a forcing,global,o,c,"${gcm}-rcp${rcp}" ru_st_MARv3.5.2_${gcm}-rcp${rcp}_2006-2100.nc
        fi
    done
done
