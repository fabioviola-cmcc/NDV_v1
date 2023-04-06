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

## Define input data directory path [default (AdriaClim released): '/data/products/ADRIACLIM_RESM/NEMO/']
#path_input = ['/data/products/ADRIACLIM_RESM/NEMO_OLD_WIND/'] # AdriaClim
#path_input = ['/data/inputs/metocean/historical/model/ocean/CMCC/CMEMS/'] # CMEMS reanalysis
#path_input = ['/data/products/ADRIACLIM_RESM/NEMO_OLD_WIND/',
#              '/data/inputs/metocean/historical/model/ocean/CMCC/CMEMS/'] # AdriaClim and CMEMS
path_input = ['/data/products/ADRIACLIM_RESM/NEMO_OLD_WIND/',
              '/data/products/ADRIACLIM_RESM/NEMO_OLD_WIND/'] # Historical and projection

## Define filename format [F=frequency, YYYYMMDD=date, G=grid]
#file_input = ["f'ADRIACLIM2_{freq}_YYYYMMDD_grid_{grid}.nc'"] # AdriaClim
#file_input = ["f'YYYYMMDD_{freq}-CMCC--{grid}-BSe2r2-BS-b20180101_re-fv08.00.nc'"] # CMEMS Black Sea
#file_input = ["f'YYYYMMDD_{freq}-CMCC--{grid}-MFSe3r1-MED-b20200901_re-sv01.00.nc'"] # CMEMS Mediterranean
#file_input = ["f'ADRIACLIM2_{freq}_YYYYMMDD_grid_{grid}.nc'",
#              "f'YYYYMMDD_{freq}-CMCC--{grid}-MFSe3r1-MED-b20200901_re-sv01.00.nc'"] # AdriaClim and CMEMS Mediterranean
file_input = ["f'ADRIACLIM2_{freq}_YYYYMMDD_grid_{grid}.nc'",
              "f'ADRIACLIM2_{freq}_YYYYMMDD_grid_{grid}.nc'"] # Historical and projection

## Define mask directory path [default (AdriaClim released): '/data/products/ADRIACLIM_RESM/NEMO/']
#path_mask = ['/data/products/ADRIACLIM_RESM/NEMO_OLD_WIND/'] # AdriaClim
#path_mask = ['/work/opa/ag35322/apuama/output/'] # CMEMS Adriatic (area_input = [39, 46, 12, 21])
#path_mask = ['/data/products/ADRIACLIM_RESM/NEMO_OLD_WIND/',
#             '/data/opa/mfs/Med_static/MFS_EAS6_STATIC_V5/NEMO_DATA0/data/'] # AdriaClim and CMEMS Adriatic
path_mask = ['/data/products/ADRIACLIM_RESM/NEMO_OLD_WIND/',
             '/data/products/ADRIACLIM_RESM/NEMO_OLD_WIND/'] # Historical and projection

## Define mask filename
#file_mask = ['mask_NEMO_AdriaClim.nc'] #AdriaClim
#file_mask = ['mesh_mask_gebco24_v8_drd_v3.nc'] #CMEMS
#file_mask = ['mask_NEMO_AdriaClim.nc','mesh_mask_gebco24_v8_drd_v3.nc'] # AdriaClim and CMEMS
file_mask = ['mask_NEMO_AdriaClim.nc','mask_NEMO_AdriaClim.nc'] # Historical and projection

## Define output directory path
path_output = '/work/opa/ag35322/apuama/output/'

## Define output subdirectory path [default if empty: YYYYMMDD_HHMM)]
subpath_output = 'test_Hist_Proj_allvars_7'

## Define directory path for intermediate NetCDF files
path_file = '/work/opa/ag35322/apuama/files/'

## Define type of run ['historical' or 'projection' (NEMO) or reanalysis (CMEMS)]
#run_input = ['historical']
#run_input = ['reanalysis']
#run_input = ['historical', 'reanalysis']
run_input = ['historical', 'projection']

## Define model output frequency ['3h' or 'day']
#freq_input = 'day'
freq_input = 'day'

## Define initial time [format: YYYYMM]
timei = ['199201', '202201']

## Define final time [format: YYYYMM]
timef = ['200112', '203112']

## Define area [format: S,N,W,E]
area_input = [39, 46, 12, 21]

## Define cmap range [TRUE for fixed cmap range (in monthly and seasonal maps) of FALSE for varying cmap range]
cmaprange_input = False

## Define variables ['T', 'S', 'SST', 'SSS', 'SSH', 'WaterFlux', 'SaltFlux', 'HeatFlux', 'MLD', 'TurboclineDepth',
## 'U', 'EIVU', 'TauX', 'V', 'EIVV', 'TauY', 'W', 'EIVW', 'VertEddyVisc', 'LatEddyDiff', 'SurfVel', 'T_VertInt']
## If analysis involves reanalysis data: ['SST', 'SSS', 'SSH','T_VertInt','SurfVel']
var_input = ['SST', 'SSS', 'SSH','T_VertInt','SurfVel']

## Define types of plots [numbered list]
# 1. Daily time series of spatial mean
# 2. Monthly time series of spatial mean
# 3. Seasonal time series of spatial mean
# 4. 2D map of mean field
# 5. 2D map of monthly mean field
# 6. 2D map of seasonal mean field
# 7. Mean year
plot_input = [7]
