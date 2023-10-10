##################################################################
##################################################################
# Nemo Diagnostics & Validation (NDV) package
# ndv_input.py
#
# Afonso Gonçalves Neto - CMCC, OPA division
# afonso.goncalves@cmcc.it
#
# Lecce, IT, December 29, 2022
##################################################################
##################################################################
## Define NEMO version (Options ERDDAP (pos-proc nemo output), 3.6 or 4.2)
nemo_version = '4.2'
## Define input data directory path [default (AdriaClim released): '/data/products/ADRIACLIM_RESM/NEMO/']
#path_input = ['/data/products/ADRIACLIM_RESM/ERDDAP/NEMO_NEW_WIND/'] # AdriaClim
#path_input = ['/data/inputs/metocean/historical/model/ocean/CMCC/CMEMS/'] # CMEMS reanalysis
#path_input = ['/data/products/ADRIACLIM_RESM/ERDDAP/NEMO_NEW_WIND/',
#              '/data/inputs/metocean/historical/model/ocean/CMCC/CMEMS/'] # AdriaClim and CMEMS
path_input = ['/work/opa/vs15521/MURAT_test/',
              '/data/inputs/metocean/historical/model/ocean/CMCC/CMEMS/'] # Historical and projection

## Define filename format [F=frequency, YYYYMMDD=date, G=grid]
#file_input = ["f'ADRIACLIM2_{freq}_YYYYMMDD_grid_{grid}.nc'"] # AdriaClim
#file_input = ["f'YYYYMMDD_{freq}-CMCC--{grid}-BSe2r2-BS-b20180101_re-fv08.00.nc'"] # CMEMS Black Sea
#file_input = ["f'YYYYMMDD_{freq}-CMCC--{grid}-MFSe3r1-MED-b20200901_re-sv01.00.nc'"] # CMEMS Mediterranean
file_input = ["f'FLAME_MED24_{freq}_YYYYMMDD_grid_{grid}.nc'",
              "f'YYYYMMDD_{freq}-CMCC--{grid}-MFSe3r1-MED-b20200901_re-sv01.00.nc'"] # AdriaClim and CMEMS Mediterranean
#file_input = ["f'ADRIACLIM2_{freq}_YYYYMMDD_grid_{grid}.nc'",
#              "f'YYYYMMDD_{freq}-CMCC--{grid}-MFSe3r1-MED-b20200901_re-sv01.00.nc'"]
#file_input = ["f'ADRIACLIM2_{freq}_YYYYMMDD_grid_{grid}.nc'",
#              "f'ADRIACLIM2_{freq}_YYYYMMDD_grid_{grid}.nc'"] # Historical and projection

## Define mask directory path [default (AdriaClim released): '/data/products/ADRIACLIM_RESM/NEMO/']
#path_mask = ['/data/products/ADRIACLIM_RESM/ERDDAP/NEMO_NEW_WIND/info/'] # AdriaClim
#path_mask = ['/data/opa/mfs/Med_static/MFS_EAS6_STATIC_V5/NEMO_DATA0/data/'] # CMEMS Adriatic (area_input = [39, 46, 12, 21])
path_mask = ['/data/products/ADRIACLIM_RESM/ERDDAP/NEMO_NEW_WIND/info/',
             '/data/opa/mfs/Med_static/MFS_EAS6_STATIC_V5/NEMO_DATA0/data/'] # AdriaClim and CMEMS Adriatic
#path_mask = ['/data/products/ADRIACLIM_RESM/NEMO_OLD_WIND/',
#             '/data/products/ADRIACLIM_RESM/NEMO_OLD_WIND/'] # Historical and projection

## Define mask filename
#file_mask = ['mask_NEMO_AdriaClim.nc'] #AdriaClim
#file_mask = ['mesh_mask_gebco24_v8_drd_v3.nc'] #CMEMS
#file_mask = ['mesh_mask.nc','mesh_mask_gebco24_v8_drd_v3.nc'] # AdriaClim and CMEMS
file_mask = ['mask_NEMO_AdriaClim.nc','mesh_mask_gebco24_v8_drd_v3.nc'] # Historical and projection

## Define output directory path
path_output = '/work/opa/vs15521/NDV_v2/output/'

## Define output subdirectory path [default if empty: YYYYMMDD_HHMM)]
subpath_output = 'test_Hist_Rean_allvars_1234567'

## Define directory path for intermediate NetCDF files
path_file = '/work/opa/vs15521/NDV_v2/files/'

## Define type of run ['historical' or 'projection' (NEMO) or reanalysis (CMEMS)]
#run_input = ['historical']
#run_input = ['projection']
#run_input = ['reanalysis']
run_input = ['historical', 'reanalysis']
#run_input = ['historical', 'projection']

## Define model output frequency ['3h' or 'day']
freq_input = 'day'

## Define initial time [format: YYYYMM]
#timei = ['199201']
#timei = ['202201']
timei = ['201001', '201001']
#timei = ['199201', '199201']

## Define final time [format: YYYYMM]
#timef = ['201112']
#timef = ['204112']
#timef = ['199212', '199212']
timef = ['201912', '201912']
#timef = ['201112', '204112']

## Define area [format: S,N,W,E]
area_input = [39, 46, 12, 21]

## Define cmap range [TRUE for fixed cmap range (in monthly and seasonal maps) of FALSE for varying cmap range]
cmaprange_input = False
## Define cmap range for the difference fields between experiments
## [TRUE for fixed cmap range (in monthly and seasonal maps) of FALSE for varying cmap range]
cmaprange_diff_input = True
## If cmaprange_diff_input = True, set the thresholds to be used in diff. maps
## in the following order ['SST', 'SSS', 'SSH','T_VertInt','SurfVel']
cmap_diff_limits=[2, 2, 0.5, 2, 1]

## Define variables ['T', 'S', 'SST', 'SSS', 'SSH', 'WaterFlux', 'SaltFlux', 'HeatFlux', 'MLD', 'TurboclineDepth',
## 'U', 'EIVU', 'TauX', 'V', 'EIVV', 'TauY', 'W', 'EIVW', 'VertEddyVisc', 'LatEddyDiff', 'SurfVel', 'T_VertInt']
## If analysis involves reanalysis data: ['SST', 'SSS', 'SSH','T_VertInt','SurfVel']
var_input = ['SST', 'SSS', 'SSH','T_VertInt'] #,'SurfVel']

## Define types of plots [numbered list]
# 1. Daily time series of spatial mean
# 2. Monthly time series of spatial mean
# 3. Seasonal time series of spatial mean
# 4. 2D map of mean field
# 5. 2D map of monthly mean field
# 6. 2D map of seasonal mean field
# 7. Mean year
plot_input = [6]
