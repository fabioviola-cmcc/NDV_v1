##################################################################
##################################################################
# Nemo Diagnostics & Validation (NDV) package
# ndv_checkinput.py
#
# Afonso Gonçalves Neto - CMCC, OPA division
# afonso.goncalves@cmcc.it
#
# Lecce, IT, January 4, 2022
##################################################################
##################################################################
## Import packages
import os
from datetime import datetime
################################################################## 

## Check input data directory path [default (AdriaClim release): '/data/products/ADRIACLIM_RESM/NEMO/']
def check_inputpath(path_input):
    for path in path_input:
        if os.path.exists(path):
            a = '    Input path OK.'
        else:
            raise ValueError('Invalid input directory path.')
    return(print(a))
        
## Check filename format [F=frequency, YYYYMMDD=date, G=grid]
def check_filename(file_input):
    file_check = ["f'ADRIACLIM2_{freq}_YYYYMMDD_grid_{grid}.nc'",
                  "f'YYYYMMDD_{freq}-CMCC--{grid}-BSe2r2-BS-b20180101_re-fv08.00.nc'",
                  "f'YYYYMMDD_{freq}-CMCC--{grid}-MFSe3r1-MED-b20200901_re-sv01.00.nc'"]
    for file in file_input:
        if file in file_check:
            a = '    Filename OK.'
        else:
            raise ValueError('Invalid filename.')
    return(print(a))

## Check mask directory path [default (AdriaClim release): '/data/products/ADRIACLIM_RESM/NEMO/']
def check_maskpath(path_mask):
    for maskpath in path_mask:
        if os.path.exists(maskpath) or maskpath == '':
            a = '    Mask path OK.'
        else:
            raise ValueError('Invalid mask directory path.')
    return(print(a))
        
## Check mask filename [default (AdriaClim release): 'mask_NEMO_AdriaClim.nc']
def check_maskfile(file_mask):
    maskfile_check = ['mask_NEMO_AdriaClim.nc','mesh_mask_gebco24_v8_drd_v3.nc']
    for maskfile in file_mask:
        if maskfile in maskfile_check:
            a = '    Mask filename OK.'
        else:
            raise ValueError('Invalid mask filename.')
    return(print(a))

## Check output directory path
def check_outputpath(path_output):
    if os.path.exists(path_output):
        a = '    Output path OK.'
    else:
        raise ValueError('Invalid output directory path.')
    return(print(a))

## Check output subdirectory path
def check_outputsubpath(subpath_output):
    if isinstance(subpath_output, str):
        a = '    Output subdirectory path OK.'
    else:
        raise ValueError('Invalid output subdirectory path.')
    return(print(a))

## Check type of run ['historical' or 'projection']
def check_run(run_input):
    run_check = ['historical', 'projection', 'reanalysis']
    for run in run_input:
        if run in run_check:
            a = '    Run OK.'
        else:
            raise ValueError('Invalid run (Should be historical or projection).')
    return(print(a))

## Check model output frequency ['3h' or 'day']
def check_freq(freq_input):
    if freq_input == '3h' or freq_input == 'day':
        a = '    Frequency OK.'
    else:
        raise ValueError('Invalid frequency (Should be 3h or day).')
    return(print(a))

## Check initial and final time [format: YYYYMM]
def check_date(timei, timef, run_input):
    if len(timei) != 6 or len(timef) != 6:
        raise ValueError('Date format is not YYYYMM.')
    if int(timei) < int(timef):
        for run in run_input:
            if run == 'historical':
                if int(timei[0:4]) >= 1992 and int(timei[0:4]) <= 2020:
                    if int(timei[4:6]) >= 1 and int(timei[4:6]) <= 12:
                        a = '    Initial date OK.'
                    else:
                        raise ValueError('Invalid initial month (Should be between 01 and 12).')
                else:
                    raise ValueError('Invalid initial year (Should be between 1992 and 2020).')
                if int(timef[0:4]) >= 1992 and int(timef[0:4]) <= 2020:
                    if int(timef[4:6]) >= 1 and int(timef[4:6]) <= 12:
                        a = '    Initial and final dates OK.'
                    else:
                        raise ValueError('Invalid final month (Should be between 01 and 12).')
                else:
                    raise ValueError('Invalid final year (Should be between 1992 and 2020).')
            elif run == 'projection':
                if int(timei[0:4]) >= 2022 and int(timei[0:4]) <= 2050:
                    if int(timei[4:6]) >= 1 and int(timei[4:6]) <= 12:
                        a = '    Initial date OK.'
                    else:
                        raise ValueError('Invalid initial month (Should be between 01 and 12).')
                else:
                    raise ValueError('Invalid initial year (Should be between 2022 and 2050).')
                if int(timef[0:4]) >= 2022 and int(timef[0:4]) <= 2050:
                    if int(timef[4:6]) >= 1 and int(timef[4:6]) <= 12:
                        a = '    Initial and final dates OK.'
                    else:
                        raise ValueError('Invalid final month (Should be between 01 and 12).')
                else:
                    raise ValueError('Invalid final year (Should be between 2022 and 2050).')
            elif run == 'reanalysis':
                if int(timei[0:4]) >= 1992 and int(timei[0:4]) <= 2020:
                    if int(timei[4:6]) >= 1 and int(timei[4:6]) <= 12:
                        a = '    Initial date OK.'
                    else:
                        raise ValueError('Invalid initial month (Should be between 01 and 12).')
                else:
                    raise ValueError('Invalid initial year (Should be between 1992 and 2020).')
                if int(timef[0:4]) >= 1992 and int(timef[0:4]) <= 2020:
                    if int(timef[4:6]) >= 1 and int(timef[4:6]) <= 12:
                        a = '    Initial and final dates OK.'
                    else:
                        raise ValueError('Invalid final month (Should be between 01 and 12).')
                else:
                    raise ValueError('Invalid final year (Should be between 1992 and 2020).')
    else:
        raise ValueError('Initial date > Final date.')
    return(print(a))

## Check area [format: S,N,W,E]
def check_area(area_input):
    if len(area_input) != 4:
        raise ValueError('Invalid lat/lon format.')
    if area_input[1] < area_input[0] or area_input[3] < area_input[2]:
        raise ValueError('Invalid lat/lon values.')
    if area_input[0] > 90 or area_input[0] < -90 or area_input[1] > 90 or area_input[1] < -90:
        raise ValueError('Invalid lat/lon values.')
        
## Check cmap range [TRUE for fixed cmap range (in monthly and seasonal maps) of FALSE for varying cmap range]
def check_cmaprange(cmaprange_input):
    if cmaprange_input == True or cmaprange_input == False:
        a = '    Initial date OK.'
    else:
        raise ValueError('Invalid cmap range format (Should be TRUE or FALSE).')
    return(print(a))

## Check variables ['T', 'S', 'SST', 'SSS', 'SSH', 'WaterFlux', 'SaltFlux', 'HeatFlux', 'MLD', 'TurboclineDepth',
## 'U', 'EIVU', 'TauX', 'V', 'EIVV', 'TauY', 'W', 'EIVW', 'VertEddyVisc', 'LatEddyDiff', 'SurfVel', 'T_VertInt']
def check_var(var_input, run_input):
    if 'reanalysis' in run_input:
        var_check = ['T', 'S', 'SST', 'SSS', 'SSH', 'SurfVel', 'T_VertInt']
    else:
        var_check = ['T', 'S', 'SST', 'SSS', 'SSH', 'WaterFlux', 'SaltFlux', 'HeatFlux', 'MLD', 'TurboclineDepth',
                     'U', 'EIVU', 'TauX', 'V', 'EIVV', 'TauY', 'W', 'EIVW', 'VertEddyVisc', 'LatEddyDiff', 
                     'SurfVel', 'T_VertInt']
    for var in var_input:
        if var in var_check:
            a = '    Variables OK.'
        else:
            raise ValueError(f'Invalid variable {var}.')
    return(print(a))

## Check plots types [numbered list]
# 1. Daily time series of spatial mean
# 2. Monthly time series of spatial mean
# 3. Annual time series of spatial mean
# 4. 2D map of mean field
# 5. 2D map of monthly mean field
# 6. 2D map of seasonal mean field
# 7. Mean year
def check_plot(timei, timef, plot_input):
    plot_check = [1, 2, 3, 4, 5, 6, 7]
    time_check = (datetime.strptime(timef, '%Y%m') - datetime.strptime(timei, '%Y%m')).days
    for plot in plot_input:
        if plot in plot_check:
            if (plot == 2 or plot == 3 or plot == 5 or plot == 6 or plot == 7) and time_check  < 699:
                raise ValueError('One or more of the selected plotting methods requires at least 2 years of data.')
            a = '    Plots OK.'
        else:
            raise ValueError(f'Invalid plot type {plot}.')
    return(print(a))
