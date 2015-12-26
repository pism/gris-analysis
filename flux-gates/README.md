Flux Gates
================================

Flux gates are given as line shape files. In QGIS, use processing/densify (QChainage should work too but is untested) to generate flux gates with equally-spaced points. Then use ```extract_profiles.py``` from [pypismtools](https://github.com/pism/pypismtools) to extract the profiles from netCDF files.